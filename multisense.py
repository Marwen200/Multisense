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
║  SUBCOMMANDS (all functionality in this single file)                         ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  python multisense.py train                                                  ║
║      --model ms_fusion --seed 3 --data_root ./data                           ║
║                                                                              ║
║  python multisense.py generate_splits                                        ║
║      --data_root ./data --output_dir ./data/splits                           ║
║                                                                              ║
║  python multisense.py attention_profile                                      ║
║      --checkpoint ./output/ms_fusion/seed_3/best.pt --data_root ./data      ║
║                                                                              ║
║  python multisense.py loeo                                                   ║
║      --data_root ./data --output_dir ./output/loeo                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  MODELS (from the paper)                                                     ║
║    MSFusionModel         — proposed (1.57 M params, 700 M FLOPs)            ║
║    SqueezeNetConcatModel — fair dual-modality baseline (1.25 M)              ║
║    LightweightCNN        — single-modality lightweight (1.98 M)              ║
║    ViTBaseline           — ViT-B/16 single-modality (86.57 M)               ║
║    VGG16Baseline         — VGG-16  single-modality (138.36 M)               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════════
# §0  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os, csv, glob, json, time, random, logging, argparse
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
CLASS_NAMES = [
    "derna_flood",       # 0
    "gaza_conflict",     # 1
    "syria_turkey_eq",   # 2
    "hurricane_harvey",  # 3
    "no_damage",         # 4
]
NUM_CLASSES  = len(CLASS_NAMES)
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

# Disaster events that can be held-out in the LOEO experiment
LOEO_EVENTS = ["derna_flood", "gaza_conflict", "syria_turkey_eq", "hurricane_harvey"]

# Attention-profile thresholds (Section 9.5)
AMP_THRESH = 0.7
SUP_THRESH = 0.3

# Balanced test set: 600 pairs per class (Section 4.4)
BALANCED_PER_CLASS = 600

_NORM = dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


# ══════════════════════════════════════════════════════════════════════════════
# §2  DATASET — HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _pad_to_square(img: Image.Image) -> Image.Image:
    """Centre-pad a PIL image with black borders to produce a square canvas."""
    w, h  = img.size
    s     = max(w, h)
    bg    = Image.new(img.mode, (s, s), 0)
    bg.paste(img, ((s - w) // 2, (s - h) // 2))
    return bg


def _collect_images(directory: str) -> list[str]:
    """Return sorted list of all image files found recursively under directory."""
    exts  = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(files)


def _make_transform(load_size: int, crop_size: int, augment: bool) -> transforms.Compose:
    """Return the appropriate image transform pipeline."""
    if augment:
        return transforms.Compose([
            transforms.Lambda(_pad_to_square),
            transforms.Resize((load_size, load_size)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomCrop((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(**_NORM),
        ])
    return transforms.Compose([
        transforms.Lambda(_pad_to_square),
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(**_NORM),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# §3  DATASET — MultiSenseDataset
# ══════════════════════════════════════════════════════════════════════════════

class MultiSenseDataset(Dataset):
    """
    Paired satellite-tile + UAV-frame dataset for the MultiSense benchmark.

    Train split : UAV frame chosen randomly at __getitem__ (re-sampled each epoch
                  — acts as implicit augmentation, Section 4.4).
    Val / Test  : UAV frame fixed at construction time via numpy.random.seed(42),
                  one seed application per class (Section 4.4).

    Two construction modes:
      A) Directory scan (default) — scans data_root/satellite/<split>/<class>/
         and data_root/uav/<split>/<class>/ for image files.
      B) pairs_csv — reads a CSV with columns: satellite_path, uav_path, label, split.
    """

    def __init__(
        self,
        data_root: str,
        split:     str  = "train",
        balanced:  bool = False,
        pairs_csv: Optional[str] = None,
        augment:   bool = True,
        load_size: int  = 228,
        crop_size: int  = 224,
    ):
        self.split    = split
        self.augment  = augment and (split == "train")
        self.tf       = _make_transform(load_size, crop_size, self.augment)
        self.pairs    = (self._from_csv(pairs_csv, split)
                         if pairs_csv else self._scan(data_root, split))
        if balanced:
            self.pairs = self._balance(self.pairs)
        print(f"  [{split}] {len(self.pairs)} pairs  (balanced={balanced})")

    # ── internal: scan directories ────────────────────────────────────────────
    def _scan(self, data_root: str, split: str) -> list[dict]:
        pairs = []
        for label, cls in enumerate(CLASS_NAMES):
            sat_dir = os.path.join(data_root, "satellite", split, cls)
            uav_dir = os.path.join(data_root, "uav",       split, cls)
            if not (os.path.isdir(sat_dir) and os.path.isdir(uav_dir)):
                continue
            sat = _collect_images(sat_dir)
            uav = _collect_images(uav_dir)
            if not sat or not uav:
                continue
            if split == "train":
                for s in sat:
                    pairs.append({"sat": s, "uav_pool": uav,
                                  "label": label, "fixed_uav": None})
            else:
                rng = np.random.default_rng(seed=42)
                assigned = uav_files = uav
                idx = rng.choice(len(uav_files), size=len(sat), replace=True)
                for i, s in enumerate(sat):
                    pairs.append({"sat": s, "uav_pool": None,
                                  "label": label, "fixed_uav": uav_files[idx[i]]})
        return pairs

    # ── internal: load from pre-built CSV ─────────────────────────────────────
    def _from_csv(self, csv_path: str, split: str) -> list[dict]:
        pairs = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                lbl = (int(row["label"]) if row["label"].isdigit()
                       else CLASS_TO_IDX[row["label"]])
                pairs.append({"sat": row["satellite_path"], "uav_pool": None,
                              "label": lbl, "fixed_uav": row["uav_path"]})
        return pairs

    # ── internal: balance to smallest class ───────────────────────────────────
    def _balance(self, pairs: list[dict]) -> list[dict]:
        from collections import defaultdict
        buckets: dict = defaultdict(list)
        for p in pairs:
            buckets[p["label"]].append(p)
        n   = min(len(v) for v in buckets.values())
        rng = np.random.default_rng(seed=42)
        out = []
        for items in buckets.values():
            out.extend([items[i] for i in rng.choice(len(items), n, replace=False)])
        return out

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        p   = self.pairs[idx]
        uav = p["fixed_uav"] or random.choice(p["uav_pool"])
        with Image.open(p["sat"]).convert("RGB") as img:
            sat_t = self.tf(img)
        with Image.open(uav).convert("RGB") as img:
            uav_t = self.tf(img)
        return {"satellite": sat_t, "uav": uav_t, "label": p["label"]}


# ══════════════════════════════════════════════════════════════════════════════
# §4  DATASET — LOEODataset  (Section 9.4)
# ══════════════════════════════════════════════════════════════════════════════

class LOEODataset(Dataset):
    """
    Wraps MultiSenseDataset to implement Leave-One-Event-Out filtering.

    held_out_only=False: excludes the held-out disaster class (training split).
    held_out_only=True : keeps ONLY the held-out disaster class (evaluation).

    No Damage tiles associated with the held-out geographic site are also
    excluded from training by checking the path for the event keyword.
    """

    def __init__(self, base: MultiSenseDataset, held_out: str,
                 held_out_only: bool = False):
        self.base      = base
        held_label     = CLASS_NAMES.index(held_out)
        nd_label       = CLASS_NAMES.index("no_damage")
        kw             = held_out.split("_")[0]   # e.g. "derna" from "derna_flood"
        self.indices   = []
        for i, p in enumerate(base.pairs):
            lbl = p["label"]
            if held_out_only:
                if lbl == held_label:
                    self.indices.append(i)
            else:
                if lbl == held_label:
                    continue
                if lbl == nd_label and kw in p["sat"].lower():
                    continue
                self.indices.append(i)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        return self.base[self.indices[idx]]


# ══════════════════════════════════════════════════════════════════════════════
# §5  BASE MODEL
# ══════════════════════════════════════════════════════════════════════════════

class BaseModel(nn.Module):
    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir

    def save(self, name: str) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.save_dir, name + ".pt"))

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)


# ══════════════════════════════════════════════════════════════════════════════
# §6  SHARED SQUEEZENET-1.1 BACKBONE
# ══════════════════════════════════════════════════════════════════════════════

class SqueezeNetBackbone(nn.Module):
    """
    SqueezeNet-1.1 feature extractor — the weight-shared Siamese backbone.

    Output: f ∈ R^{512}  (D=512, from fire8 global average pool).
    Parameters: 1.24 M.   FLOPs per pass: ~349 M MACs.
    """
    D = 512

    def __init__(self):
        super().__init__()
        sqn = models.squeezenet1_1(
            weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1
        )
        self.features = sqn.features          # conv1 + fire2-fire8
        self.pool     = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)          # (N, 512)


# ══════════════════════════════════════════════════════════════════════════════
# §7  PROPOSED MODEL — MSFusionModel  (Equations 4-9, Section 6.2)
# ══════════════════════════════════════════════════════════════════════════════

class MSFusionModel(BaseModel):
    """
    MultiSense Attention Fusion Model — main contribution.

    Eq. 4  f_c   = f_sat + f_uav                        (element-wise add)
    Eq. 5  f_tilde= ReLU(BN(W_v · f_c))   W_v  ∈ R^{d×D}
    Eq. 6  alpha  = σ(W_attn · f_c)        W_attn ∈ R^{d×D}
    Eq. 7  m      = alpha ⊙ f_tilde
    Eq. 8  h      = ReLU(BN(W_r · m))      W_r   ∈ R^{d×d}
    Eq. 9  y      = W_cls · h              W_cls  ∈ R^{C×d}

    Weights: Xavier uniform.  Biases: zero.
    Total: 1.57 M params (1.24 M backbone + 0.33 M fusion).  700 M FLOPs.
    Returns: (logits, alpha)  — alpha exported for interpretability.
    """

    def __init__(self, save_dir: str, num_class: int = NUM_CLASSES,
                 D: int = 512, d: int = 256):
        super().__init__(save_dir)
        self.backbone  = SqueezeNetBackbone()

        # Fusion layers
        self.proj_bn   = nn.BatchNorm1d(d)
        self.W_v       = nn.Linear(D, d)

        self.W_attn    = nn.Linear(D, d)

        self.refine_bn = nn.BatchNorm1d(d)
        self.W_r       = nn.Linear(d, d)

        self.W_cls     = nn.Linear(d, num_class)
        self.dropout   = nn.Dropout(p=0.3)

        for layer in [self.W_v, self.W_attn, self.W_r, self.W_cls]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: tuple) -> tuple:
        sat, uav  = x
        f_sat     = self.backbone(sat)            # (N, 512)
        f_uav     = self.backbone(uav)            # (N, 512)

        f_c       = f_sat + f_uav                 # Eq. 4
        f_tilde   = F.relu(self.proj_bn(self.W_v(f_c)))     # Eq. 5
        alpha     = torch.sigmoid(self.W_attn(f_c))          # Eq. 6
        m         = alpha * f_tilde                           # Eq. 7
        h         = F.relu(self.refine_bn(self.W_r(m)))      # Eq. 8
        logits    = self.W_cls(self.dropout(h))               # Eq. 9

        return logits, alpha    # alpha shape (N, 256)


# ══════════════════════════════════════════════════════════════════════════════
# §8  FAIR BASELINE — SqueezeNetConcatModel  (Section 7)
# ══════════════════════════════════════════════════════════════════════════════

class SqueezeNetConcatModel(BaseModel):
    """
    Same Siamese SqueezeNet-1.1 backbone; concatenation replaces attention.
    Isolates the 0.70 pp attention contribution.  1.25 M params.
    Returns: (logits, None)
    """
    def __init__(self, save_dir: str, num_class: int = NUM_CLASSES, D: int = 512):
        super().__init__(save_dir)
        self.backbone   = SqueezeNetBackbone()
        self.classifier = nn.Linear(2 * D, num_class)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: tuple) -> tuple:
        sat, uav  = x
        h         = torch.cat([self.backbone(sat), self.backbone(uav)], dim=1)
        return self.classifier(h), None


# ══════════════════════════════════════════════════════════════════════════════
# §9  SINGLE-MODALITY — LightweightCNN  (Section 6.1, L-CNN)
# ══════════════════════════════════════════════════════════════════════════════

class _DSConv(nn.Module):
    """Depthwise-Separable Convolution block used inside each L-CNN block."""
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class LightweightCNN(BaseModel):
    """
    5-block lightweight CNN — single-modality satellite baseline.
    Channels {32,64,128,256,256}.  MaxPool after blocks 1-3.
    GAP → Dropout(0.3) → Linear(256 → num_class).
    1.98 M params, 1.37 B FLOPs.  Returns: (logits, None)
    """
    def __init__(self, save_dir: str, num_class: int = NUM_CLASSES):
        super().__init__(save_dir)
        ch = [3, 32, 64, 128, 256, 256]
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch[i], ch[i+1], 3, padding=1, bias=False),
                nn.BatchNorm2d(ch[i+1]),
                nn.ReLU(inplace=True),
                _DSConv(ch[i+1]),
            ) for i in range(5)
        ])
        self.pool       = nn.MaxPool2d(2, stride=2)
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(256, num_class)

    def forward(self, x: tuple) -> tuple:
        h, _ = x
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if i < 3:
                h = self.pool(h)
        h = self.gap(h).view(h.size(0), -1)
        return self.classifier(self.dropout(h)), None


# ══════════════════════════════════════════════════════════════════════════════
# §10  SINGLE-MODALITY — ViT-B/16 and VGG-16 baselines  (Section 7)
# ══════════════════════════════════════════════════════════════════════════════

class ViTBaseline(BaseModel):
    """ViT-B/16.  86.57 M params, 17.58 G FLOPs.  Returns: (logits, None)"""
    def __init__(self, save_dir: str, num_class: int = NUM_CLASSES):
        super().__init__(save_dir)
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        vit.heads = nn.Linear(vit.hidden_dim, num_class)
        self.vit = vit

    def forward(self, x: tuple) -> tuple:
        return self.vit(x[0]), None


class VGG16Baseline(BaseModel):
    """VGG-16 + BN.  138.36 M params, 15.47 G FLOPs.  Returns: (logits, None)"""
    def __init__(self, save_dir: str, num_class: int = NUM_CLASSES):
        super().__init__(save_dir)
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        vgg.classifier[6] = nn.Linear(4096, num_class)
        self.vgg = vgg

    def forward(self, x: tuple) -> tuple:
        return self.vgg(x[0]), None


# ══════════════════════════════════════════════════════════════════════════════
# §11  MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, type] = {
    "ms_fusion":         MSFusionModel,
    "squeezenet_concat": SqueezeNetConcatModel,
    "lcnn":              LightweightCNN,
    "vit":               ViTBaseline,
    "vgg16":             VGG16Baseline,
}


# ══════════════════════════════════════════════════════════════════════════════
# §12  TRAINER  (Section 8 hyperparameters)
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Unified training / validation / prediction loop for all MultiSense models.

    Hyperparameters (fixed per Section 8 of the paper):
      SGD, lr=2e-3, momentum=0.9 | batch_size=16
      ReduceLROnPlateau(factor=0.1, patience=5)
      Early stopping patience=10
      Gradient clipping max-norm=1.0
      Mixed-precision AMP (GPU only)
    """

    def __init__(self, model, train_loader, val_loader, test_loader,
                 save_dir, device="cuda", patience=10, display=50):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.save_dir     = save_dir
        self.device       = device
        self.patience     = patience
        self.display      = display
        os.makedirs(save_dir, exist_ok=True)

        self.loss_fn  = nn.CrossEntropyLoss()
        self.opt      = optim.SGD(model.parameters(), lr=2e-3, momentum=0.9)
        self.sched    = optim.lr_scheduler.ReduceLROnPlateau(
                            self.opt, factor=0.1, patience=5, verbose=True)
        self.scaler   = (torch.cuda.amp.GradScaler()
                         if device != "cpu" else None)

    def _xy(self, batch):
        sat = batch["satellite"].to(self.device)
        uav = batch["uav"].to(self.device)
        y   = batch["label"].to(self.device)
        return (sat, uav), y

    def train(self, max_epochs: int) -> dict:
        best_loss = float("inf")
        no_imp    = 0
        with open(os.path.join(self.save_dir, "times.txt"), "w") as tf:
            for ep in range(max_epochs):
                t0 = time.time()
                self.model.train()
                tot_loss = cor = tot = dl = dc = dt = 0
                for bi, batch in enumerate(
                    tqdm(self.train_loader, desc=f"Ep {ep:>3}", leave=False)
                ):
                    x, y = self._xy(batch)
                    self.opt.zero_grad()
                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            logits, _ = self.model(x)
                            loss      = self.loss_fn(logits, y)
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.opt)
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.opt)
                        self.scaler.update()
                    else:
                        logits, _ = self.model(x)
                        loss      = self.loss_fn(logits, y)
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.opt.step()
                    bs   = x[0].size(0)
                    bc   = (logits.argmax(1) == y).sum().item()
                    tot_loss += loss.item(); cor += bc; tot += bs
                    dl += loss.item();       dc += bc; dt += bs
                    if (bi + 1) % self.display == 0:
                        logging.info("  [%d/%d] loss=%.4f acc=%.4f",
                                     bi+1, len(self.train_loader),
                                     dl/dt, dc/dt)
                        dl = dc = dt = 0

                logging.info("Ep %d acc=%.4f loss=%.6f", ep, cor/tot, tot_loss/tot)
                self.model.save(f"ckpt_{ep:04d}")
                vl, va = self.validate()
                self.sched.step(vl)
                if vl < best_loss:
                    best_loss = vl; no_imp = 0
                    self.model.save("best")
                    logging.info("  ✓ val_loss=%.6f", best_loss)
                else:
                    no_imp += 1
                    if no_imp >= self.patience:
                        logging.info("Early stop at epoch %d", ep); break
                tf.write(f"Ep {ep}: {time.time()-t0:.1f}s\n")

        bp = os.path.join(self.save_dir, "best.pt")
        if os.path.exists(bp):
            self.model.load(bp)
        return self.predict()

    def validate(self) -> tuple[float, float]:
        self.model.eval()
        tl = cor = tot = 0
        with torch.no_grad():
            for batch in self.val_loader:
                x, y    = self._xy(batch)
                lg, _   = self.model(x)
                tl     += self.loss_fn(lg, y).item()
                cor    += (lg.argmax(1) == y).sum().item()
                tot    += x[0].size(0)
        vl, va = tl/tot, cor/tot
        logging.info("  val loss=%.6f acc=%.4f", vl, va)
        return vl, va

    def predict(self) -> dict:
        self.model.eval()
        preds: list[int]  = []
        labels: list[int] = []
        probs:  list      = []
        alphas: list      = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Test", leave=False):
                x, y     = self._xy(batch)
                lg, attn = self.model(x)
                pr = torch.softmax(lg, 1).cpu().numpy()
                preds.extend(lg.argmax(1).cpu().tolist())
                labels.extend(y.cpu().tolist())
                probs.extend(pr)
                if attn is not None:
                    alphas.append(attn.cpu().numpy())

        with open(os.path.join(self.save_dir, "prediction.csv"), "w") as f:
            f.writelines(f"{p}\n" for p in preds)
        if alphas:
            np.save(os.path.join(self.save_dir, "attention_profiles.npy"),
                    np.vstack(alphas))
            np.save(os.path.join(self.save_dir, "test_labels.npy"),
                    np.array(labels))

        m = compute_metrics(preds, labels,
                            np.array(probs) if probs else None)
        logging.info("Test acc=%.4f F1=%.4f", m["accuracy"], m["f1"])
        return m


# ══════════════════════════════════════════════════════════════════════════════
# §13  METRICS  (Section 8)
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(preds, labels, probs=None) -> dict:
    """Accuracy, macro P/R/F1, OvR AUC, per-class breakdown."""
    n = len(labels)
    pc = {}
    mp = mr = mf = 0.0
    for i, cls in enumerate(CLASS_NAMES):
        tp = sum(p == i and l == i for p, l in zip(preds, labels))
        fp = sum(p == i and l != i for p, l in zip(preds, labels))
        fn = sum(p != i and l == i for p, l in zip(preds, labels))
        pr = tp / (tp + fp + 1e-9)
        rc = tp / (tp + fn + 1e-9)
        f1 = 2 * pr * rc / (pr + rc + 1e-9)
        pc[cls] = {"precision": pr, "recall": rc, "f1": f1}
        mp += pr; mr += rc; mf += f1

    auc = None
    if _SKLEARN and probs is not None:
        try:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except Exception:
            pass

    return {
        "accuracy":  sum(p == l for p, l in zip(preds, labels)) / n,
        "precision": mp / NUM_CLASSES,
        "recall":    mr / NUM_CLASSES,
        "f1":        mf / NUM_CLASSES,
        "auc":       auc,
        "per_class": pc,
    }


def print_metrics(m: dict, title: str = "") -> None:
    """Pretty-print metric dict to stdout."""
    print(f"\n{'='*65}\n  {title}\n{'='*65}")
    print(f"  Accuracy : {m['accuracy']*100:.2f}%  Macro-F1: {m['f1']*100:.2f}%"
          f"  AUC: {m['auc']*100:.2f}%" if m['auc'] else
          f"  Accuracy : {m['accuracy']*100:.2f}%  Macro-F1: {m['f1']*100:.2f}%")
    print(f"  {'Class':<25} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"  {'─'*52}")
    for cls, s in m["per_class"].items():
        print(f"  {cls:<25} {s['precision']:>8.4f} {s['recall']:>8.4f} {s['f1']:>8.4f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# §14  TRAIN SUBCOMMAND
# ══════════════════════════════════════════════════════════════════════════════

def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_seed(model_key: str, seed: int, args) -> dict:
    """Train one model for one seed; return test metrics."""
    _seed_everything(seed)
    save_dir = os.path.join(args.output_dir, model_key, f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(save_dir, "run.log"), level=logging.INFO,
        format="%(asctime)s %(message)s", force=True,
    )
    # Datasets
    train_set = MultiSenseDataset(
        args.data_root, "train",
        balanced=getattr(args, "balanced_train", False),
        pairs_csv=getattr(args, "pairs_csv", None), augment=True)
    val_set = MultiSenseDataset(
        args.data_root, "val",
        pairs_csv=getattr(args, "pairs_csv", None), augment=False)
    test_set = MultiSenseDataset(
        args.data_root, "test",
        balanced=getattr(args, "balanced_test", False),
        pairs_csv=getattr(args, "pairs_csv", None), augment=False)

    nw = getattr(args, "num_workers", 2)
    train_loader = DataLoader(train_set, 16, shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_set,   32, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_set,  32, shuffle=False, num_workers=nw, pin_memory=True)

    model = MODEL_REGISTRY[model_key](save_dir=save_dir).to(args.device)
    if getattr(args, "checkpoint", None):
        model.load(args.checkpoint)

    trainer = Trainer(model, train_loader, val_loader, test_loader,
                      save_dir=save_dir, device=args.device)

    if getattr(args, "eval", False):
        trainer.validate()
        return trainer.predict()
    return trainer.train(args.max_epochs)


def cmd_train(args) -> None:
    """Train subcommand — runs one or more models over one or more seeds."""
    keys = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    seeds = ([args.seed] if hasattr(args, "seed") and args.seed is not None
             else list(range(args.seeds)))

    for key in keys:
        print(f"\n{'█'*65}\n  Model: {key.upper()}\n{'█'*65}")
        all_m = []
        for seed in seeds:
            print(f"\n  ── Seed {seed} ──")
            m = run_one_seed(key, seed, args)
            all_m.append(m)
            print_metrics(m, title=f"{key}  seed={seed}")

        if len(all_m) > 1:
            print(f"\n  {'─'*40}")
            print(f"  {key.upper()} — mean ± std over {len(all_m)} seeds")
            for metric in ["accuracy", "f1", "precision", "recall"]:
                vals = [m[metric]*100 for m in all_m]
                print(f"    {metric:<12}: {np.mean(vals):.2f} ± {np.std(vals):.2f} %")
            auc_vals = [m["auc"]*100 for m in all_m if m["auc"] is not None]
            if auc_vals:
                print(f"    {'auc (OvR)':<12}: "
                      f"{np.mean(auc_vals):.2f} ± {np.std(auc_vals):.2f} %")


# ══════════════════════════════════════════════════════════════════════════════
# §15  GENERATE_SPLITS SUBCOMMAND  (Section 4.4)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_generate_splits(args) -> None:
    """
    Build 70/15/15 stratified group split with seed 42 and write:
      imbalanced_test_pairs.json  (all test tiles, fixed UAV pairing)
      balanced_test_pairs.json    (600 pairs per class)
      pairs_all.csv               (master CSV, all splits)
      train_val_test_split.json   (per-class file lists)
    """
    os.makedirs(args.output_dir, exist_ok=True)
    split_registry: dict = {}
    imbalanced: list[dict] = []
    balanced:   list[dict] = []
    csv_rows:   list[dict] = []

    RATIOS = (0.70, 0.15, 0.15)

    def split_list(lst, seed=42):
        rng = random.Random(seed)
        lst = list(lst); rng.shuffle(lst)
        n   = len(lst)
        nv  = max(1, round(n * RATIOS[1]))
        nt  = max(1, round(n * RATIOS[2]))
        return lst[:n-nv-nt], lst[n-nv-nt:n-nt], lst[n-nt:]

    def fixed_pairs(sat, uav, label, split, n_bal, seed=42):
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(len(uav), size=len(sat), replace=True)
        all_p = [{"satellite_path": sat[i], "uav_path": uav[idx[i]],
                  "label": label, "split": split}
                 for i in range(len(sat))]
        if len(sat) <= n_bal:
            bal = list(all_p)
        else:
            sel = rng.choice(len(sat), size=n_bal, replace=False)
            bal = [all_p[i] for i in sorted(sel)]
        return all_p, bal

    for label, cls in enumerate(CLASS_NAMES):
        sat_dir = os.path.join(args.data_root, "satellite", cls)
        uav_dir = os.path.join(args.data_root, "uav",       cls)
        if not os.path.isdir(sat_dir) or not os.path.isdir(uav_dir):
            print(f"  [SKIP] {cls} not found"); continue

        sat = _collect_images(sat_dir)
        uav = _collect_images(uav_dir)
        if not sat or not uav:
            print(f"  [SKIP] {cls} empty"); continue

        tr_s, vl_s, te_s = split_list(sat, args.seed)
        tr_u, vl_u, te_u = split_list(uav, args.seed)

        split_registry[cls] = {"label": label,
                                "train_sat": tr_s, "val_sat": vl_s, "test_sat": te_s,
                                "train_uav": tr_u, "val_uav": vl_u, "test_uav": te_u}
        print(f"  {cls}: {len(tr_s)}/{len(vl_s)}/{len(te_s)} train/val/test sat tiles")

        vl_all, _ = fixed_pairs(vl_s, vl_u, label, "val",  args.balanced_per_class, args.seed)
        te_all, te_bal = fixed_pairs(te_s, te_u, label, "test", args.balanced_per_class, args.seed)
        imbalanced.extend(te_all)
        balanced.extend(te_bal)
        for s in tr_s:
            csv_rows.append({"satellite_path": s, "uav_path": "", "label": label, "split": "train"})
        csv_rows.extend(vl_all)
        csv_rows.extend(te_all)

    def save_json(obj, name):
        p = os.path.join(args.output_dir, name)
        with open(p, "w") as f: json.dump(obj, f, indent=2)
        print(f"  {name}  →  {len(obj) if isinstance(obj, list) else '...'} entries")

    save_json(split_registry, "train_val_test_split.json")
    save_json(imbalanced,     "imbalanced_test_pairs.json")
    save_json(balanced,       "balanced_test_pairs.json")

    csv_p = os.path.join(args.output_dir, "pairs_all.csv")
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["satellite_path","uav_path","label","split"])
        w.writeheader(); w.writerows(csv_rows)
    print(f"  pairs_all.csv  →  {len(csv_rows)} rows")
    print(f"\n✓  Splits written to {args.output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# §16  ATTENTION_PROFILE SUBCOMMAND  (Section 9.5, Table 12)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_attention_profile(args) -> None:
    """
    Compute class-conditional mean attention profiles  ᾱ_c = (1/|T_c|)·Σα(x).
    Outputs Table 12 (Amplified / Suppressed / Neutral %) to stdout and CSV.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Collect alpha vectors ─────────────────────────────────────────────────
    if getattr(args, "npy_path", None):
        alpha_mat = np.load(args.npy_path)
        lbl_arr   = np.load(args.labels_npy)
    else:
        model = MSFusionModel(save_dir=args.output_dir).to(args.device)
        model.load(args.checkpoint); model.eval()
        test_set    = MultiSenseDataset(args.data_root, "test", augment=False,
                                        pairs_csv=getattr(args,"pairs_csv",None))
        test_loader = DataLoader(test_set, 64, shuffle=False,
                                 num_workers=getattr(args,"num_workers",2))
        alphas: list = []; lbls: list = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Collecting α"):
                sat = batch["satellite"].to(args.device)
                uav = batch["uav"].to(args.device)
                _, alpha = model((sat, uav))
                alphas.append(alpha.cpu().numpy())
                lbls.extend(batch["label"].tolist())
        alpha_mat = np.vstack(alphas)
        lbl_arr   = np.array(lbls)
        np.save(os.path.join(args.output_dir, "attention_profiles.npy"), alpha_mat)
        np.save(os.path.join(args.output_dir, "test_labels.npy"), lbl_arr)

    # ── Per-class profiles ────────────────────────────────────────────────────
    profiles: dict = {}
    for idx, cls in enumerate(CLASS_NAMES):
        mask = lbl_arr == idx
        if not mask.any(): continue
        mu  = alpha_mat[mask].mean(axis=0)
        amp = (mu > AMP_THRESH).sum()
        sup = (mu < SUP_THRESH).sum()
        neu = len(mu) - amp - sup
        profiles[cls] = {
            "n": int(mask.sum()),
            "mean_alpha":     mu.tolist(),
            "amplified_pct":  round(amp/len(mu)*100, 1),
            "suppressed_pct": round(sup/len(mu)*100, 1),
            "neutral_pct":    round(neu/len(mu)*100, 1),
        }

    # ── Print Table 12 ────────────────────────────────────────────────────────
    print(f"\n  Table 12 — Class-Conditional Attention Profile (256 channels)")
    print(f"  {'Class':<25} {'Amplified%':>11} {'Suppressed%':>12} {'Neutral%':>10}")
    print(f"  {'─'*62}")
    for cls in CLASS_NAMES:
        if cls not in profiles: continue
        s = profiles[cls]
        print(f"  {cls:<25} {s['amplified_pct']:>11.1f} "
              f"{s['suppressed_pct']:>12.1f} {s['neutral_pct']:>10.1f}")

    # ── Verify theoretical predictions ────────────────────────────────────────
    if "no_damage" in profiles:
        nd_sup = profiles["no_damage"]["suppressed_pct"]
        others = [profiles[c]["suppressed_pct"] for c in CLASS_NAMES
                  if c != "no_damage" and c in profiles]
        print(f"\n  P1 (No Damage highest suppression): "
              f"{'✓' if nd_sup > max(others) else '✗'}  ND={nd_sup:.1f}%  max_others={max(others):.1f}%")

    # ── Save outputs ──────────────────────────────────────────────────────────
    with open(os.path.join(args.output_dir, "class_profiles.json"), "w") as f:
        json.dump(profiles, f, indent=2)
    csv_p = os.path.join(args.output_dir, "attention_profile_table.csv")
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class","n","amplified_pct","suppressed_pct","neutral_pct"])
        w.writeheader()
        for cls in CLASS_NAMES:
            if cls not in profiles: continue
            s = profiles[cls]
            w.writerow({"class":cls,"n":s["n"],"amplified_pct":s["amplified_pct"],
                        "suppressed_pct":s["suppressed_pct"],"neutral_pct":s["neutral_pct"]})
    print(f"\n✓  Attention profiles → {args.output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# §17  LOEO SUBCOMMAND  (Section 9.4, Table 11)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_loeo(args) -> None:
    """
    Leave-One-Event-Out generalisation experiment.
    Trains MS Fusion on 3 events, evaluates on the held-out 4th.
    Produces Table 11.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    results: dict = {}

    for held_out in LOEO_EVENTS:
        print(f"\n  ── Held-out: {held_out.upper()} ──")
        _seed_everything(args.seed)
        run_dir = os.path.join(args.output_dir, f"loeo_{held_out}")
        os.makedirs(run_dir, exist_ok=True)

        raw_tr = MultiSenseDataset(args.data_root, "train",
                                   pairs_csv=getattr(args,"pairs_csv",None), augment=True)
        raw_vl = MultiSenseDataset(args.data_root, "val",
                                   pairs_csv=getattr(args,"pairs_csv",None), augment=False)
        raw_te = MultiSenseDataset(args.data_root, "test",
                                   pairs_csv=getattr(args,"pairs_csv",None), augment=False)

        tr = LOEODataset(raw_tr, held_out, held_out_only=False)
        vl = LOEODataset(raw_vl, held_out, held_out_only=False)
        te = LOEODataset(raw_te, held_out, held_out_only=True)
        print(f"    Train:{len(tr)}  Val:{len(vl)}  Held-out:{len(te)}")

        nw = getattr(args, "num_workers", 2)
        tr_l = DataLoader(tr, 16, shuffle=True,  num_workers=nw, pin_memory=True)
        vl_l = DataLoader(vl, 32, shuffle=False, num_workers=nw, pin_memory=True)
        te_l = DataLoader(te, 32, shuffle=False, num_workers=nw, pin_memory=True)

        model    = MSFusionModel(save_dir=run_dir).to(args.device)
        loss_fn  = nn.CrossEntropyLoss()
        opt      = optim.SGD(model.parameters(), lr=2e-3, momentum=0.9)
        sched    = optim.lr_scheduler.ReduceLROnPlateau(opt, 0.1, patience=5)
        scaler   = (torch.cuda.amp.GradScaler()
                    if args.device != "cpu" else None)

        if not getattr(args, "eval_only", False):
            best = float("inf"); no_imp = 0
            for ep in range(args.max_epochs):
                model.train()
                for batch in tqdm(tr_l, desc=f"  Ep{ep:>3}", leave=False):
                    sat = batch["satellite"].to(args.device)
                    uav = batch["uav"].to(args.device)
                    y   = batch["label"].to(args.device)
                    opt.zero_grad()
                    if scaler:
                        with torch.cuda.amp.autocast():
                            lg, _ = model((sat, uav))
                            loss  = loss_fn(lg, y)
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(opt); scaler.update()
                    else:
                        lg, _ = model((sat, uav))
                        loss  = loss_fn(lg, y)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                model.eval()
                vl_loss = vc = vt = 0
                with torch.no_grad():
                    for b in vl_l:
                        sat = b["satellite"].to(args.device)
                        uav = b["uav"].to(args.device)
                        y   = b["label"].to(args.device)
                        lg, _ = model((sat, uav))
                        vl_loss += loss_fn(lg, y).item()
                        vc += (lg.argmax(1)==y).sum().item(); vt += y.size(0)
                vl_avg = vl_loss / vt
                sched.step(vl_avg)
                if vl_avg < best:
                    best = vl_avg; no_imp = 0; model.save("best")
                else:
                    no_imp += 1
                    if no_imp >= 10: break

        bp = os.path.join(run_dir, "best.pt")
        if os.path.exists(bp): model.load(bp)

        # Evaluate on held-out event tiles only
        model.eval()
        preds: list[int]  = []
        lbls:  list[int]  = []
        probs: list       = []
        with torch.no_grad():
            for b in te_l:
                sat = b["satellite"].to(args.device)
                uav = b["uav"].to(args.device)
                y   = b["label"]
                lg, _ = model((sat, uav))
                probs.extend(torch.softmax(lg,1).cpu().numpy())
                preds.extend(lg.argmax(1).cpu().tolist())
                lbls.extend(y.tolist())

        m = compute_metrics(preds, lbls,
                            np.array(probs) if probs else None)
        results[held_out] = {"accuracy": round(m["accuracy"]*100, 2),
                             "macro_f1": round(m["f1"]*100, 2),
                             "n_tiles":  len(lbls)}
        print(f"    Acc={results[held_out]['accuracy']:.2f}%  "
              f"F1={results[held_out]['macro_f1']:.2f}%")

    # ── Table 11 ──────────────────────────────────────────────────────────────
    print(f"\n  Table 11 — Leave-One-Event-Out Results")
    print(f"  {'Held-out Event':<25} {'Tiles':>6} {'Acc%':>8} {'Macro-F1%':>11}")
    print(f"  {'─'*54}")
    acc_v = []; f1_v = []
    for ev, r in results.items():
        print(f"  {ev:<25} {r['n_tiles']:>6} {r['accuracy']:>8.2f} {r['macro_f1']:>11.2f}")
        acc_v.append(r["accuracy"]); f1_v.append(r["macro_f1"])
    print(f"  {'─'*54}")
    print(f"  {'LOEO Mean':<25} {'':>6} {np.mean(acc_v):>8.2f} {np.mean(f1_v):>11.2f}")
    print(f"  (random baseline = {100/NUM_CLASSES:.1f}%)\n")

    results["LOEO Mean"] = {"accuracy": round(np.mean(acc_v),2),
                             "macro_f1": round(np.mean(f1_v),2)}
    with open(os.path.join(args.output_dir,"loeo_results.json"),"w") as f:
        json.dump(results, f, indent=2)
    csv_p = os.path.join(args.output_dir,"loeo_results_table.csv")
    with open(csv_p,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["event","acc","macro_f1"])
        w.writeheader()
        for ev, r in results.items():
            w.writerow({"event":ev,"acc":r["accuracy"],"macro_f1":r["macro_f1"]})
    print(f"✓  LOEO results → {args.output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# §18  ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def get_args() -> argparse.Namespace:
    root = argparse.ArgumentParser(
        description="MultiSense — all functionality in one file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = root.add_subparsers(dest="cmd", required=True)

    # ── Common args shared by several subcommands ─────────────────────────────
    def _common(p):
        p.add_argument("--data_root",   default="./data")
        p.add_argument("--output_dir",  default="./output")
        p.add_argument("--pairs_csv",   default=None)
        p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--num_workers", type=int, default=2)
        return p

    # ── train ──────────────────────────────────────────────────────────────────
    p_train = _common(sub.add_parser("train", help="Train a model"))
    p_train.add_argument("--model", default="ms_fusion",
                         choices=list(MODEL_REGISTRY.keys()) + ["all"])
    p_train.add_argument("--seed",  type=int, default=None,
                         help="Run a single specific seed.")
    p_train.add_argument("--seeds", type=int, default=5,
                         help="Run seeds 0..N-1 (used when --seed is not set).")
    p_train.add_argument("--max_epochs", type=int, default=300)
    p_train.add_argument("--balanced_train", action="store_true")
    p_train.add_argument("--balanced_test",  action="store_true")
    p_train.add_argument("--eval",       action="store_true")
    p_train.add_argument("--checkpoint", default=None)

    # ── generate_splits ───────────────────────────────────────────────────────
    p_split = _common(sub.add_parser("generate_splits", help="Build fixed pair files"))
    p_split.add_argument("--seed",              type=int, default=42)
    p_split.add_argument("--balanced_per_class",type=int, default=BALANCED_PER_CLASS)

    # ── attention_profile ─────────────────────────────────────────────────────
    p_attn = _common(sub.add_parser("attention_profile", help="Compute Table 12"))
    p_attn.add_argument("--checkpoint", default=None)
    p_attn.add_argument("--npy_path",   default=None,
                        help="Load pre-saved attention_profiles.npy")
    p_attn.add_argument("--labels_npy", default=None)

    # ── loeo ──────────────────────────────────────────────────────────────────
    p_loeo = _common(sub.add_parser("loeo", help="Leave-One-Event-Out experiment"))
    p_loeo.add_argument("--max_epochs", type=int, default=300)
    p_loeo.add_argument("--seed",       type=int, default=42)
    p_loeo.add_argument("--eval_only",  action="store_true")

    return root.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# §19  MAIN DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = get_args()
    dispatch = {
        "train":             cmd_train,
        "generate_splits":   cmd_generate_splits,
        "attention_profile": cmd_attention_profile,
        "loeo":              cmd_loeo,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
