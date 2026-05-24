#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         MultiSense — Complete Single-File Implementation                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Paper : "MultiSense: A Knowledge-Guided Multimodal Benchmark and Attention  ║
║           Fusion Framework for Multi-Disaster Detection in Smart Cities"      ║
║  Journal: The Visual Computer (Springer), 2025                               ║
║  Authors: Marwen Bouabid · Mohamed Farah                                     ║
║  Code   : https://github.com/Marwen200/Multisense                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  MODELS IMPLEMENTED (all from the paper)                                     ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  Proposed:                                                                   ║
║    MSFusionModel          — Siamese SqueezeNet-1.1 + channel-attention       ║
║                             fusion (1.57 M params, 700 M FLOPs)              ║
║  Baselines:                                                                  ║
║    SqueezeNetConcatModel  — same backbone, concatenation head (1.25 M)       ║
║    LightweightCNN         — 5-block lightweight single-modality (1.98 M)     ║
║    ViTBaseline            — ViT-B/16 single-modality (86.57 M params)        ║
║    VGG16Baseline          — VGG-16   single-modality (138.36 M params)       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USAGE                                                                       ║
║  ─────                                                                       ║
║  # Reproduce the main MS-Fusion result (Table 4 of the paper)               ║
║  python multisense.py --model ms_fusion --data_root ./data                   ║
║                                                                              ║
║  # Train every model sequentially over 5 seeds                              ║
║  python multisense.py --model all --seeds 5                                  ║
║                                                                              ║
║  # Evaluate a saved checkpoint                                               ║
║  python multisense.py --model ms_fusion --eval                               ║
║      --checkpoint ./output/ms_fusion/seed_3/best.pt                         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DATASET LAYOUT  (set --data_root to the folder containing these)           ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  data_root/                                                                  ║
║    satellite/                                                                ║
║      train/  val/  test/                                                     ║
║        derna_flood/      *.jpg (or *.png)                                    ║
║        gaza_conflict/                                                        ║
║        syria_turkey_eq/                                                      ║
║        hurricane_harvey/                                                     ║
║        no_damage/                                                            ║
║    uav/                                                                      ║
║      train/  val/  test/                                                     ║
║        (same class sub-folders as satellite)                                 ║
║                                                                              ║
║  Alternatively supply --pairs_csv pointing to a CSV with columns:           ║
║    satellite_path, uav_path, label, split                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DEPENDENCIES                                                                ║
║    torch==2.0.1  torchvision==0.15.2  Pillow  numpy  tqdm  scikit-learn     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════════
# §0  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os
import csv
import glob
import time
import random
import logging
import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


# ══════════════════════════════════════════════════════════════════════════════
# §1  GLOBAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Five disaster classes from the MultiSense benchmark (Section 5.1 of the paper)
CLASS_NAMES = [
    "derna_flood",       # 0 — Derna dam collapse, Libya 2023
    "gaza_conflict",     # 1 — Gaza conflict, Palestine 2023
    "syria_turkey_eq",   # 2 — Syria-Turkey earthquake 2023
    "hurricane_harvey",  # 3 — Hurricane Harvey, USA 2017
    "no_damage",         # 4 — pre-event control imagery (same 4 sites)
]
NUM_CLASSES  = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ImageNet normalisation used for all pretrained backbones (images in [-1, 1])
_NORM_MEAN = (0.5, 0.5, 0.5)
_NORM_STD  = (0.5, 0.5, 0.5)


# ══════════════════════════════════════════════════════════════════════════════
# §2  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class MultiSenseDataset(Dataset):
    """
    PyTorch Dataset for the MultiSense benchmark.

    Each sample is a pair (satellite_tile, uav_frame) with a shared disaster
    class label.  Two construction modes are supported:

    Mode A — directory scan (default):
        Scans ``<data_root>/satellite/<split>/<class>/`` and
              ``<data_root>/uav/<split>/<class>/`` for image files.
        Test-set pairs are fixed at construction via numpy.random.seed(42)
        (one seed application per class, matching the paper protocol).
        Training pairs are re-sampled randomly each epoch.

    Mode B — CSV file (``pairs_csv`` is not None):
        Reads a CSV with columns: satellite_path, uav_path, label, split.
        Useful when you already have the fixed-pair JSON/CSV from the
        published dataset release.

    Args:
        data_root:    Root directory containing ``satellite/`` and ``uav/``.
        split:        ``'train'``, ``'val'``, or ``'test'``.
        balanced:     If True, cap each class to min-class size (balanced test set).
        pairs_csv:    Optional path to a pre-built pairs CSV.
        augment:      If True, apply training augmentations (only for train split).
        load_size:    Resize shorter side to this value before cropping (228).
        crop_size:    Final crop fed to the network (224).
    """

    def __init__(
        self,
        data_root:  str,
        split:      str  = "train",
        balanced:   bool = False,
        pairs_csv:  Optional[str] = None,
        augment:    bool = True,
        load_size:  int  = 228,
        crop_size:  int  = 224,
    ):
        self.split     = split
        self.balanced  = balanced
        self.augment   = augment and (split == "train")

        # ── Image transform pipelines ─────────────────────────────────────────
        # Training: mild augmentation (horizontal flip p=0.2 + random crop)
        # Val/Test: deterministic centre crop
        if self.augment:
            self.transform = transforms.Compose([
                transforms.Lambda(self._pad_to_square),
                transforms.Resize((load_size, load_size)),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(_NORM_MEAN, _NORM_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Lambda(self._pad_to_square),
                transforms.Resize((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(_NORM_MEAN, _NORM_STD),
            ])

        # ── Build sample pairs ────────────────────────────────────────────────
        if pairs_csv is not None:
            self.pairs = self._load_from_csv(pairs_csv, split)
        else:
            self.pairs = self._scan_directory(data_root, split)

        if balanced:
            self.pairs = self._balance(self.pairs)

        print(f"[MultiSenseDataset] {split}: {len(self.pairs)} pairs loaded.")

    # ── Static helper: pad image to square ───────────────────────────────────
    @staticmethod
    def _pad_to_square(img: Image.Image) -> Image.Image:
        """Centre-pad a PIL image with black borders to make it square."""
        w, h  = img.size
        side  = max(w, h)
        bg    = Image.new(img.mode, (side, side), 0)
        bg.paste(img, ((side - w) // 2, (side - h) // 2))
        return bg

    # ── Build pairs by scanning directory tree ─────────────────────────────
    def _scan_directory(self, data_root: str, split: str) -> list[dict]:
        """
        Scan <data_root>/satellite/<split>/<class>/ and
             <data_root>/uav/<split>/<class>/
        and create satellite-UAV pairs.

        Training:  each satellite tile is associated with a random same-class
                   UAV frame (re-drawn each epoch via __getitem__).
        Val / Test: pairs are fixed at construction time using
                    numpy.random.seed(42) — one seed application per class,
                    matching the paper's protocol exactly.
        """
        pairs = []
        sat_root = os.path.join(data_root, "satellite", split)
        uav_root = os.path.join(data_root, "uav",       split)

        for label, class_name in enumerate(CLASS_NAMES):
            sat_dir = os.path.join(sat_root, class_name)
            uav_dir = os.path.join(uav_root, class_name)

            if not os.path.isdir(sat_dir) or not os.path.isdir(uav_dir):
                continue  # class not present in this split

            sat_files = sorted(
                glob.glob(os.path.join(sat_dir, "*.jpg")) +
                glob.glob(os.path.join(sat_dir, "*.png")) +
                glob.glob(os.path.join(sat_dir, "*.jpeg"))
            )
            uav_files = sorted(
                glob.glob(os.path.join(uav_dir, "*.jpg")) +
                glob.glob(os.path.join(uav_dir, "*.png")) +
                glob.glob(os.path.join(uav_dir, "*.jpeg"))
            )

            if not sat_files or not uav_files:
                continue

            if split == "train":
                # Training: store paths; UAV frame chosen randomly at __getitem__
                for sat in sat_files:
                    pairs.append({"sat": sat, "uav_pool": uav_files,
                                  "label": label, "fixed_uav": None})
            else:
                # Val / Test: fix pairs with seed-42 RNG (one application per class)
                rng = np.random.default_rng(seed=42)
                uav_assigned = [
                    uav_files[i % len(uav_files)]
                    for i in rng.choice(
                        len(uav_files), size=len(sat_files), replace=True
                    )
                ]
                for sat, uav in zip(sat_files, uav_assigned):
                    pairs.append({"sat": sat, "uav_pool": None,
                                  "label": label, "fixed_uav": uav})

        return pairs

    # ── Build pairs from a CSV file ───────────────────────────────────────────
    def _load_from_csv(self, csv_path: str, split: str) -> list[dict]:
        """Load pre-built pairs from a CSV with columns: sat, uav, label, split."""
        pairs = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                label = (
                    int(row["label"]) if row["label"].isdigit()
                    else CLASS_TO_IDX[row["label"]]
                )
                pairs.append({
                    "sat":       row["satellite_path"],
                    "uav_pool":  None,
                    "label":     label,
                    "fixed_uav": row["uav_path"],
                })
        return pairs

    # ── Balance classes to smallest class size ────────────────────────────────
    def _balance(self, pairs: list[dict]) -> list[dict]:
        """Downsample each class to the smallest class size (balanced eval)."""
        from collections import defaultdict
        buckets: dict[int, list] = defaultdict(list)
        for p in pairs:
            buckets[p["label"]].append(p)
        min_count = min(len(v) for v in buckets.values())
        balanced  = []
        rng       = np.random.default_rng(seed=42)
        for items in buckets.values():
            idx = rng.choice(len(items), size=min_count, replace=False)
            balanced.extend([items[i] for i in idx])
        return balanced

    # ── Dataset interface ─────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        p = self.pairs[index]

        # Select UAV frame
        if p["fixed_uav"] is not None:
            uav_path = p["fixed_uav"]          # val / test: deterministic
        else:
            uav_path = random.choice(p["uav_pool"])  # train: random same-class

        with Image.open(p["sat"]).convert("RGB")  as img:
            sat_tensor = self.transform(img)
        with Image.open(uav_path).convert("RGB") as img:
            uav_tensor = self.transform(img)

        return {
            "satellite": sat_tensor,   # (3, 224, 224)
            "uav":       uav_tensor,   # (3, 224, 224)
            "label":     p["label"],   # int in [0, 4]
        }


# ══════════════════════════════════════════════════════════════════════════════
# §3  BASE MODEL (save / load utilities)
# ══════════════════════════════════════════════════════════════════════════════

class BaseModel(nn.Module):
    """Shared save / load interface for all MultiSense models."""

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir

    def save(self, filename: str) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.state_dict(),
                   os.path.join(self.save_dir, filename + ".pt"))

    def load(self, filepath: str) -> None:
        sd = torch.load(filepath, map_location="cpu")
        self.load_state_dict(sd, strict=False)


# ══════════════════════════════════════════════════════════════════════════════
# §4  SHARED SQUEEZENET-1.1 BACKBONE
# ══════════════════════════════════════════════════════════════════════════════

class SqueezeNetBackbone(nn.Module):
    """
    SqueezeNet-1.1 feature extractor — the weight-shared Siamese backbone
    used in both MSFusionModel and SqueezeNetConcatModel.

    Architecture:
        Input (3, 224, 224)
        → SqueezeNet-1.1 features (conv1 + fire2–fire8)
        → AdaptiveAvgPool2d(1)
        → Flatten
        → f ∈ R^{512}          (D = 512 as per Section 6.1)

    The original 1000-class classifier head is discarded.
    Pretrained ImageNet weights are loaded.  The RGB satellite composite
    (Bands 6/4/2) and UAV frames are both 3-channel inputs normalised to
    [-1, 1], directly compatible with these weights.

    Parameters: 1.24 M (as reported in Table 15 of the paper).
    FLOPs per forward pass: ~349 M MACs (torchinfo, 224×224 input).
    """

    D = 512  # output feature dimension

    def __init__(self):
        super().__init__()
        sqnet = models.squeezenet1_1(
            weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1
        )
        # Keep only the convolutional feature layers (up to fire8)
        # fire8 outputs 512 channels; the classifier head is dropped.
        self.features = sqnet.features
        self.pool     = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3, 224, 224)
        Returns:
            f: (N, 512) — flattened GAP output
        """
        x = self.features(x)   # (N, 512, H, W)
        x = self.pool(x)       # (N, 512, 1, 1)
        return x.view(x.size(0), -1)   # (N, 512)


# ══════════════════════════════════════════════════════════════════════════════
# §5  PROPOSED MODEL — MS FUSION  (Section 6.2 of the paper)
# ══════════════════════════════════════════════════════════════════════════════

class MSFusionModel(BaseModel):
    """
    MultiSense Attention Fusion Model — the paper's main contribution.

    Architecture (Equations 3-9 in Section 6.2):

        Backbone (Siamese, weight-shared):
            f_sat = SqueezeNet(r_sat),  f_uav = SqueezeNet(r_uav),
            f ∈ R^D,  D = 512

        Feature Combination (Eq. 4):
            f_c = f_sat + f_uav              ∈ R^D
            (element-wise addition — symmetric, order-invariant)

        Feature Projection (Eq. 5):
            f̃ = ReLU(BN(W_v · f_c + b_v)),   W_v ∈ R^{d×D},  d = 256

        Channel Attention Mask (Eq. 6-7):
            α = σ(W_attn · f_c + b_attn),    W_attn ∈ R^{d×D}
            m = α ⊙ f̃                         ∈ R^d

        Feature Refinement (Eq. 8):
            h = ReLU(BN(W_r · m + b_r)),      W_r ∈ R^{d×d}

        Classification (Eq. 9):
            y = W_cls · h + b_cls,             W_cls ∈ R^{C×d}

    Weight initialisation:  Xavier uniform for all W matrices;
                            zero for all bias vectors (Section 6.2).
    Total parameters: 1.57 M  (1.24 M backbone + 0.33 M fusion layers).
    Total FLOPs: ~700 M MACs  (2 × 349 M backbone + ≤0.5 M fusion).
    Memory: 6.3 MB  (float32).

    Returns:
        (logits, alpha)
          logits: (N, C) — class logits
          alpha:  (N, d) — per-channel attention weights (for interpretability)
    """

    def __init__(self, save_dir: str, num_class: int = NUM_CLASSES,
                 D: int = 512, d: int = 256):
        super().__init__(save_dir)
        self.backbone = SqueezeNetBackbone()

        # ── Fusion layers ─────────────────────────────────────────────────────
        # Feature projection: f̃ = ReLU(BN(W_v · f_c))
        self.proj_bn  = nn.BatchNorm1d(d)
        self.W_v      = nn.Linear(D, d)

        # Channel attention mask: α = σ(W_attn · f_c)
        self.W_attn   = nn.Linear(D, d)

        # Feature refinement: h = ReLU(BN(W_r · m))
        self.refine_bn = nn.BatchNorm1d(d)
        self.W_r       = nn.Linear(d, d)

        # Classification head
        self.W_cls     = nn.Linear(d, num_class)

        # Dropout for regularisation (applied to classifier input)
        self.dropout   = nn.Dropout(p=0.3)

        # Xavier uniform init + zero biases (as stated in Section 6.2)
        self._init_fusion_weights()

    def _init_fusion_weights(self) -> None:
        """Xavier uniform initialisation for all fusion layer weights."""
        for layer in [self.W_v, self.W_attn, self.W_r, self.W_cls]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        satellite, uav = x

        # ── Siamese feature extraction (weight-shared backbone) ───────────────
        f_sat = self.backbone(satellite)   # (N, 512)
        f_uav = self.backbone(uav)         # (N, 512)

        # ── Feature Combination (Eq. 4) ───────────────────────────────────────
        f_c = f_sat + f_uav                # (N, 512) element-wise sum

        # ── Feature Projection (Eq. 5) ────────────────────────────────────────
        f_tilde = F.relu(self.proj_bn(self.W_v(f_c)))    # (N, 256)

        # ── Channel Attention Mask (Eq. 6-7) ─────────────────────────────────
        alpha = torch.sigmoid(self.W_attn(f_c))           # (N, 256)  ← α
        m     = alpha * f_tilde                            # (N, 256)  ← m = α ⊙ f̃

        # ── Feature Refinement (Eq. 8) ────────────────────────────────────────
        h = F.relu(self.refine_bn(self.W_r(m)))           # (N, 256)

        # ── Classification (Eq. 9) ────────────────────────────────────────────
        logits = self.W_cls(self.dropout(h))               # (N, C)

        return logits, alpha   # return alpha for interpretability analysis


# ══════════════════════════════════════════════════════════════════════════════
# §6  FAIR DUAL-MODALITY BASELINE — SqueezeNet-Concat  (Section 7)
# ══════════════════════════════════════════════════════════════════════════════

class SqueezeNetConcatModel(BaseModel):
    """
    SqueezeNet-Concat Baseline — architecturally fair dual-modality comparison.

    Identical Siamese SqueezeNet-1.1 backbone.  Instead of the channel-attention
    fusion module, the two feature vectors are simply concatenated and fed to a
    single linear classifier without intermediate normalisation.

    This baseline isolates the contribution of the attention mechanism:
    the only architectural difference from MSFusionModel is the absence of
    the projection → attention mask → refinement pipeline.

    Architecture:
        f_sat, f_uav ∈ R^D  (D = 512)
        h = Concat(f_sat, f_uav)  ∈ R^{2D}   (1024-dim)
        y = W_c · h + b_c,         W_c ∈ R^{C×2D}

    Parameters: 1.25 M  (1.24 M backbone + 5.1 K linear head).
    FLOPs: ~700 M MACs.
    Memory: 5.0 MB.

    Returns: (logits, None)
    """

    def __init__(self, save_dir: str, num_class: int = NUM_CLASSES, D: int = 512):
        super().__init__(save_dir)
        self.backbone   = SqueezeNetBackbone()
        # Direct linear classifier — no intermediate normalisation (Section 7)
        self.classifier = nn.Linear(2 * D, num_class)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: tuple) -> tuple[torch.Tensor, None]:
        satellite, uav = x
        f_sat   = self.backbone(satellite)          # (N, 512)
        f_uav   = self.backbone(uav)                # (N, 512)
        h       = torch.cat([f_sat, f_uav], dim=1)  # (N, 1024)
        logits  = self.classifier(h)                # (N, C)
        return logits, None


# ══════════════════════════════════════════════════════════════════════════════
# §7  SINGLE-MODALITY BASELINE — Lightweight CNN  (Section 6.1, L-CNN)
# ══════════════════════════════════════════════════════════════════════════════

class _DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution block used inside each L-CNN block.

    Factorises a standard 3×3 conv into:
      Depthwise:  Conv2d(C, C, 3, padding=1, groups=C)  — spatial filter
      Pointwise:  Conv2d(C, C, 1)                        — channel mixing
      BatchNorm + ReLU
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1,
                            groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class LightweightCNN(BaseModel):
    """
    Lightweight CNN — single-modality satellite-only baseline.

    Architecture (Section 6.1, "The paper also evaluates a Lightweight CNN"):

        Five blocks, each containing:
          • Conv2d(in_ch, out_ch, 3×3, stride 1, padding 1) + BN + ReLU
          • DepthwiseSeparableConv(out_ch)                  (for efficiency)

        Channel widths:  {32, 64, 128, 256, 256}
        MaxPool(2×2, stride 2) applied after blocks 1, 2, 3.
        Global Average Pooling → 256-dim feature vector.
        Dropout(p=0.3) → Linear(256 → num_class).

    Parameters: 1.98 M
    FLOPs: 1.37 B MACs (torchinfo, 224×224 input)

    Receives only the satellite tile.
    Returns: (logits, None)
    """

    def __init__(self, save_dir: str, num_class: int = NUM_CLASSES):
        super().__init__(save_dir)
        channels = [3, 32, 64, 128, 256, 256]   # in → block outputs

        self.blocks = nn.ModuleList()
        for i in range(5):
            in_ch  = channels[i]
            out_ch = channels[i + 1]
            block  = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                _DepthwiseSeparableConv(out_ch),
            )
            self.blocks.append(block)

        # MaxPool applied after blocks 0, 1, 2  (i.e. after blocks 1, 2, 3 in
        # 1-based indexing used in the paper)
        self.pool    = nn.MaxPool2d(2, stride=2)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(256, num_class)

    def forward(self, x: tuple) -> tuple[torch.Tensor, None]:
        satellite, _ = x   # ignore UAV — single-modality model
        h = satellite
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i < 3:          # MaxPool after blocks 1, 2, 3 (0-indexed: 0,1,2)
                h = self.pool(h)
        h      = self.gap(h)
        h      = self.flatten(h)
        logits = self.classifier(self.dropout(h))
        return logits, None


# ══════════════════════════════════════════════════════════════════════════════
# §8  SINGLE-MODALITY BASELINES — ViT-B/16 and VGG-16  (Section 7)
# ══════════════════════════════════════════════════════════════════════════════

class ViTBaseline(BaseModel):
    """
    Vision Transformer baseline — single-modality (satellite tile only).

    Uses ViT-B/16 pretrained on ImageNet-21k (HuggingFace) or torchvision's
    vit_b_16.  The 1000-class head is replaced by a Linear(768, num_class) layer.

    Paper figures (Table 15):
        17.58 G FLOPs,  86.57 M params,  346.3 MB memory.

    Receives only the satellite tile.
    Returns: (logits, None)
    """

    def __init__(self, save_dir: str, num_class: int = NUM_CLASSES):
        super().__init__(save_dir)
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        # Replace the classification head with our own
        vit.heads = nn.Linear(vit.hidden_dim, num_class)
        self.vit  = vit

    def forward(self, x: tuple) -> tuple[torch.Tensor, None]:
        satellite, _ = x
        return self.vit(satellite), None


class VGG16Baseline(BaseModel):
    """
    VGG-16 baseline — single-modality (satellite tile only).

    Pretrained VGG-16 with BatchNorm; the final FC layer is replaced by
    Linear(4096, num_class).

    Paper figures (Table 15):
        15.47 G FLOPs,  138.36 M params,  553.4 MB memory.

    Receives only the satellite tile.
    Returns: (logits, None)
    """

    def __init__(self, save_dir: str, num_class: int = NUM_CLASSES):
        super().__init__(save_dir)
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        # Replace the 1000-class classifier tail
        vgg.classifier[6] = nn.Linear(4096, num_class)
        self.vgg = vgg

    def forward(self, x: tuple) -> tuple[torch.Tensor, None]:
        satellite, _ = x
        return self.vgg(satellite), None


# ══════════════════════════════════════════════════════════════════════════════
# §9  MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, type] = {
    "ms_fusion":         MSFusionModel,
    "squeezenet_concat": SqueezeNetConcatModel,
    "lcnn":              LightweightCNN,
    "vit":               ViTBaseline,
    "vgg16":             VGG16Baseline,
}


# ══════════════════════════════════════════════════════════════════════════════
# §10  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    predictions: list[int],
    labels:      list[int],
    probs:       Optional[np.ndarray] = None,
) -> dict:
    """
    Compute all metrics reported in the paper (Section 8):
      - Joint accuracy
      - Macro-averaged Precision, Recall, F1
      - Macro one-vs-rest AUC (requires scikit-learn and probability scores)

    Args:
        predictions: Predicted class indices (length N).
        labels:      Ground-truth class indices (length N).
        probs:       Softmax probability matrix (N, C) for AUC computation.

    Returns:
        Dict with keys: 'accuracy', 'precision', 'recall', 'f1', 'auc',
                        'per_class' (dict class_name → p/r/f1).
    """
    n = len(labels)
    per_class = {}
    total_p = total_r = total_f1 = 0.0

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        tp = sum(p == cls_idx and l == cls_idx for p, l in zip(predictions, labels))
        fp = sum(p == cls_idx and l != cls_idx for p, l in zip(predictions, labels))
        fn = sum(p != cls_idx and l == cls_idx for p, l in zip(predictions, labels))

        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)

        per_class[cls_name] = {"precision": prec, "recall": rec, "f1": f1}
        total_p  += prec
        total_r  += rec
        total_f1 += f1

    nc = NUM_CLASSES
    macro_p  = total_p  / nc
    macro_r  = total_r  / nc
    macro_f1 = total_f1 / nc
    accuracy = sum(p == l for p, l in zip(predictions, labels)) / n

    auc = None
    if _SKLEARN and probs is not None:
        try:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except Exception:
            pass

    return {
        "accuracy":  accuracy,
        "precision": macro_p,
        "recall":    macro_r,
        "f1":        macro_f1,
        "auc":       auc,
        "per_class": per_class,
    }


def print_metrics(metrics: dict, model_name: str = "") -> None:
    """Pretty-print metric dict to stdout."""
    header = f"\n{'=' * 70}\n  {model_name}  Results\n{'=' * 70}"
    print(header)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro-F1 : {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    if metrics["auc"] is not None:
        print(f"  AUC (OvR): {metrics['auc']:.4f}")
    print()
    print(f"  {'Class':<30} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"  {'-'*58}")
    for cls, s in metrics["per_class"].items():
        print(f"  {cls:<30} {s['precision']:>8.4f} {s['recall']:>8.4f} {s['f1']:>8.4f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# §11  TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Unified training loop for all MultiSense models.

    Training protocol (Section 8 of the paper):
      - SGD, initial lr = 2e-3, momentum = 0.9
      - Batch size = 16
      - ReduceLROnPlateau (factor = 0.1, patience = 5)
      - Early stopping: patience = 10 epochs
      - Mixed-precision (AMP) when CUDA is available
      - Gradient clipping (max-norm = 1.0)

    All models return (logits, auxiliary) where auxiliary is None or the
    attention weight vector alpha for MSFusionModel.

    Args:
        model:        Any BaseModel subclass.
        train_loader: DataLoader for training split.
        val_loader:   DataLoader for validation split.
        test_loader:  DataLoader for test split.
        save_dir:     Directory for checkpoint files and logs.
        device:       Torch device string.
        patience:     Early-stopping patience (default 10, per paper).
        display:      Log every N batches (default 50).
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        test_loader:  DataLoader,
        save_dir:     str,
        device:       str  = "cuda",
        patience:     int  = 10,
        display:      int  = 50,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.save_dir     = save_dir
        self.device       = device
        self.patience     = patience
        self.display      = display

        os.makedirs(save_dir, exist_ok=True)

        # Loss function: CrossEntropyLoss (standard for classification)
        self.loss_fn  = nn.CrossEntropyLoss()

        # SGD with momentum=0.9 (Section 8)
        self.optimizer = optim.SGD(
            model.parameters(), lr=2e-3, momentum=0.9
        )

        # ReduceLROnPlateau: factor=0.1, patience=5 (Section 8)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.1, patience=5, verbose=True
        )

        # Mixed-precision GradScaler for GPU training
        self.scaler: Optional[torch.cuda.amp.GradScaler] = (
            torch.cuda.amp.GradScaler()
            if device != "cpu" and hasattr(torch.cuda.amp, "GradScaler")
            else None
        )

    # ── Helper: move batch to device ──────────────────────────────────────────
    def _to_device(self, batch: dict) -> tuple:
        sat    = batch["satellite"].to(self.device)
        uav    = batch["uav"].to(self.device)
        labels = batch["label"].to(self.device)
        return (sat, uav), labels

    # ── Training loop ─────────────────────────────────────────────────────────
    def train(self, max_epochs: int) -> dict:
        """
        Train with early stopping.

        Returns the full metrics dict from the best checkpoint evaluated on
        the test set.
        """
        best_val_loss  = float("inf")
        no_improve     = 0
        timing_path    = os.path.join(self.save_dir, "epoch_times.txt")

        with open(timing_path, "w") as tf:
            for epoch in range(max_epochs):
                t0 = time.time()
                self.model.train()

                total_loss = correct = total = 0
                disp_loss  = disp_correct = disp_total = 0

                for batch_idx, batch in enumerate(
                    tqdm(self.train_loader, desc=f"Epoch {epoch:>3}", leave=False)
                ):
                    x, y = self._to_device(batch)
                    self.optimizer.zero_grad()

                    # Forward + backward
                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            logits, _ = self.model(x)
                            loss      = self.loss_fn(logits, y)
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        logits, _ = self.model(x)
                        loss      = self.loss_fn(logits, y)
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()

                    bs             = x[0].size(0)
                    batch_correct  = (logits.argmax(1) == y).sum().item()
                    total_loss    += loss.item()
                    correct       += batch_correct
                    total         += bs
                    disp_loss     += loss.item()
                    disp_correct  += batch_correct
                    disp_total    += bs

                    if (batch_idx + 1) % self.display == 0:
                        logging.info(
                            "  [%d/%d] loss=%.4f acc=%.4f",
                            batch_idx + 1, len(self.train_loader),
                            disp_loss / disp_total, disp_correct / disp_total,
                        )
                        disp_loss = disp_correct = disp_total = 0

                # End-of-epoch
                train_acc  = correct    / total
                train_loss = total_loss / total
                logging.info("Epoch %d | train_acc=%.4f  train_loss=%.6f",
                             epoch, train_acc, train_loss)

                self.model.save(f"checkpoint_{epoch:04d}")

                # Validation + scheduler step
                val_loss, val_acc = self.validate()
                self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve    = 0
                    self.model.save("best")
                    logging.info("  ✓ Best val_loss=%.6f  val_acc=%.4f",
                                 best_val_loss, val_acc)
                else:
                    no_improve += 1
                    logging.info("  No improvement %d/%d",
                                 no_improve, self.patience)
                    if no_improve >= self.patience:
                        logging.info("Early stopping at epoch %d", epoch)
                        break

                tf.write(f"Epoch {epoch}: {time.time()-t0:.1f}s\n")

        # Load best checkpoint and evaluate on test set
        best_path = os.path.join(self.save_dir, "best.pt")
        if os.path.exists(best_path):
            self.model.load(best_path)

        return self.predict()

    # ── Validation loop ───────────────────────────────────────────────────────
    def validate(self) -> tuple[float, float]:
        """Returns (mean_loss, accuracy) on the validation set."""
        self.model.eval()
        total_loss = correct = total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x, y      = self._to_device(batch)
                logits, _ = self.model(x)
                loss      = self.loss_fn(logits, y)
                total_loss += loss.item()
                correct    += (logits.argmax(1) == y).sum().item()
                total      += x[0].size(0)

        return total_loss / total, correct / total

    # ── Test / prediction loop ────────────────────────────────────────────────
    def predict(self) -> dict:
        """
        Run inference on the fixed imbalanced test set.

        Returns the full metrics dict (accuracy, macro-P/R/F1, AUC, per-class).
        Also writes prediction.csv and attention_profiles.npy to save_dir.
        """
        self.model.eval()
        all_preds:   list[int]   = []
        all_labels:  list[int]   = []
        all_probs:   list        = []
        all_alphas:  list        = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Test", leave=False):
                x, y        = self._to_device(batch)
                logits, attn = self.model(x)

                probs   = torch.softmax(logits, dim=1).cpu().numpy()
                indices = logits.argmax(1).cpu().tolist()

                all_preds.extend(indices)
                all_labels.extend(y.cpu().tolist())
                all_probs.extend(probs)

                # Collect attention weights for interpretability (MS Fusion only)
                if attn is not None:
                    all_alphas.append(attn.cpu().numpy())

        # Save predictions
        pred_path = os.path.join(self.save_dir, "prediction.csv")
        with open(pred_path, "w") as f:
            f.writelines(f"{p}\n" for p in all_preds)

        # Save attention profiles if available
        if all_alphas:
            alpha_matrix = np.vstack(all_alphas)  # (N_test, 256)
            np.save(os.path.join(self.save_dir, "attention_profiles.npy"),
                    alpha_matrix)

        probs_arr = np.array(all_probs)
        metrics   = compute_metrics(all_preds, all_labels, probs_arr)

        logging.info("Test accuracy: %.4f  Macro-F1: %.4f",
                     metrics["accuracy"], metrics["f1"])
        if metrics["auc"]:
            logging.info("AUC (OvR): %.4f", metrics["auc"])

        return metrics


# ══════════════════════════════════════════════════════════════════════════════
# §12  MULTI-SEED RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_one_seed(
    model_key:   str,
    seed:        int,
    args,
    data_root:   str,
) -> dict:
    """
    Set a random seed, build a fresh model + dataloaders, train, and return metrics.

    Five seeds are used to compute mean ± std as reported in Table 4 of the paper.
    """
    # Reproducibility: fix all RNG sources
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    save_dir = os.path.join(args.output_dir, model_key, f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(save_dir, "run.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    # ── Datasets and loaders ─────────────────────────────────────────────────
    # Training: dynamic random pairing (re-sampled each epoch per the paper)
    # Val: fixed pairs, full imbalanced
    # Test: fixed pairs, imbalanced (3812 pairs for the 5-class benchmark)
    train_set = MultiSenseDataset(data_root, "train",
                                  balanced=args.balanced_train,
                                  pairs_csv=args.pairs_csv,
                                  augment=True)
    val_set   = MultiSenseDataset(data_root, "val",
                                  pairs_csv=args.pairs_csv, augment=False)
    test_set  = MultiSenseDataset(data_root, "test",
                                  balanced=args.balanced_test,
                                  pairs_csv=args.pairs_csv, augment=False)

    train_loader = DataLoader(train_set, batch_size=16,   # batch_size=16 per paper
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=32,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=32,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────────
    ModelClass = MODEL_REGISTRY[model_key]
    model      = ModelClass(save_dir=save_dir, num_class=NUM_CLASSES)
    model      = model.to(args.device)

    if args.checkpoint:
        model.load(args.checkpoint)
        print(f"  Loaded checkpoint: {args.checkpoint}")

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        save_dir=save_dir,
        device=args.device,
        patience=10,    # fixed at 10 as per Section 6.1
        display=50,
    )

    if args.eval:
        _, _ = trainer.validate()
        return trainer.predict()
    else:
        return trainer.train(max_epochs=args.max_epochs)


# ══════════════════════════════════════════════════════════════════════════════
# §13  ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MultiSense — Multimodal Disaster Detection (single-file)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model selection ───────────────────────────────────────────────────────
    p.add_argument(
        "--model",
        default="ms_fusion",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help=(
            "Which model to train. 'all' runs every model in MODEL_REGISTRY. "
            "ms_fusion is the paper's main contribution; squeezenet_concat is "
            "the fair dual-modality baseline; lcnn/vit/vgg16 are single-modality."
        ),
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    p.add_argument("--data_root",   default="./data",
                   help="Root containing satellite/ and uav/ sub-directories.")
    p.add_argument("--pairs_csv",   default=None,
                   help="Optional CSV with pre-built pairs (columns: "
                        "satellite_path, uav_path, label, split).")
    p.add_argument("--balanced_train", action="store_true",
                   help="Downsample train set to balanced classes.")
    p.add_argument("--balanced_test",  action="store_true",
                   help="Use balanced test set (600 tiles/class, as in Table 4).")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--max_epochs", default=300,  type=int,
                   help="Maximum training epochs (early stopping may terminate earlier).")
    p.add_argument("--seeds",      default=5,    type=int,
                   help="Number of independent random seeds (5 in the paper).")
    p.add_argument("--output_dir", default="./output",
                   help="Root directory for checkpoints and logs.")

    # ── Evaluation ────────────────────────────────────────────────────────────
    p.add_argument("--eval",       action="store_true",
                   help="Skip training; evaluate --checkpoint on the test set.")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a .pt checkpoint to load before train/eval.")

    # ── Runtime ───────────────────────────────────────────────────────────────
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available()
                                            else "cpu")
    p.add_argument("--num_workers", default=2, type=int)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# §14  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Entry point.

    Runs ``seeds`` independent training runs for the chosen model(s)
    and prints mean ± std across seeds for all reported metrics,
    matching the reporting convention in Table 4 of the paper.
    """
    args      = get_args()
    data_root = args.data_root

    # Which models to train
    models_to_run = (
        list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    )

    for model_key in models_to_run:
        print(f"\n{'█' * 70}")
        print(f"  Model: {model_key.upper()}")
        print(f"{'█' * 70}\n")

        seed_metrics: list[dict] = []

        for seed in range(args.seeds):
            print(f"  ── Seed {seed} / {args.seeds - 1} ──")
            m = run_one_seed(model_key, seed, args, data_root)
            seed_metrics.append(m)
            print_metrics(m, model_name=f"{model_key} seed={seed}")

        # ── Aggregate: mean ± std over seeds (Table 4 format) ─────────────
        if len(seed_metrics) > 1:
            for key in ["accuracy", "f1", "precision", "recall"]:
                vals = [m[key] for m in seed_metrics]
                print(f"  {key:>10}: {np.mean(vals)*100:.2f} ± {np.std(vals)*100:.2f} %")
            auc_vals = [m["auc"] for m in seed_metrics if m["auc"] is not None]
            if auc_vals:
                print(f"  {'auc (OvR)':>10}: "
                      f"{np.mean(auc_vals)*100:.2f} ± {np.std(auc_vals)*100:.2f} %")

        print()


if __name__ == "__main__":
    main()
