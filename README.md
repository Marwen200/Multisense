# MultiSense: A Knowledge-Guided Multimodal Benchmark and Attention Fusion Framework for Multi-Disaster Detection in Smart Cities

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch 2.0.1](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg)](https://pytorch.org)
[![CUDA 11.8](https://img.shields.io/badge/CUDA-11.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![The Visual Computer](https://img.shields.io/badge/Journal-The%20Visual%20Computer-blueviolet.svg)](https://link.springer.com/journal/371)

> **Published in:** *The Visual Computer* — Springer, 2025  
> **Authors:** Marwen Bouabid · Mohamed Farah  
> **Institution:** ENSI, RIADI Laboratory LR99ES26, University of Manouba, Tunisia  
> **Code DOI:** https://doi.org/10.5281/zenodo.XXXXXXX

---

## Overview

**MultiSense** is an end-to-end framework for multi-disaster detection in smart cities that
fuses 🛰️ PlanetScope satellite tiles with 🚁 crowdsourced UAV frames via a lightweight
channel-attention fusion architecture.

**The entire implementation — all models, datasets, training, evaluation, attention analysis,
and LOEO experiment — lives in a single file: `multisense.py`.**

---

## Repository Structure

```
Multisense/
├── multisense.py      ← SINGLE IMPLEMENTATION FILE (all code here)
├── train.py           ← 3-line shim: python train.py --model ms_fusion --seed 3
├── train_all.sh       ← Reproduce all paper results (calls multisense.py only)
├── requirements.txt
└── README.md
```

---

## Quick Start

### Install

```bash
git clone https://github.com/Marwen200/Multisense.git
cd Multisense
pip install -r requirements.txt
```

### Step 1 — Generate fixed splits and pair files (once)

```bash
python multisense.py generate_splits \
    --data_root ./data \
    --output_dir ./data/splits
```

Produces:
- `imbalanced_test_pairs.json` — all test tiles with fixed UAV pairing (seed 42)
- `balanced_test_pairs.json`   — 600 pairs per class = 3 000 total
- `pairs_all.csv`              — master CSV for all splits
- `train_val_test_split.json`  — per-class file lists

### Step 2 — Train a single model

```bash
# Proposed model, representative run (seed 3)
python multisense.py train --model ms_fusion --seed 3 --data_root ./data

# Equivalently via the response-letter wrapper:
python train.py --model ms_fusion --seed 3 --data_root ./data

# All models, 5 seeds each:
python multisense.py train --model all --seeds 5 --data_root ./data
```

### Step 3 — Reproduce ALL paper results

```bash
bash train_all.sh
# Override defaults:
DATA_ROOT=/my/data OUTPUT_DIR=/my/output SEEDS=1 bash train_all.sh
```

### Step 4 — Evaluate a saved checkpoint

```bash
python multisense.py train --model ms_fusion --seed 3 \
    --eval --checkpoint ./output/ms_fusion/seed_3/best.pt \
    --data_root ./data
```

### Step 5 — Attention profiles (Table 12)

```bash
python multisense.py attention_profile \
    --checkpoint ./output/ms_fusion/seed_3/best.pt \
    --data_root  ./data \
    --output_dir ./output/ms_fusion/seed_3
```

### Step 6 — LOEO experiment (Table 11)

```bash
python multisense.py loeo \
    --data_root  ./data \
    --output_dir ./output/loeo
```

---

## All Subcommands

```
python multisense.py <subcommand> [options]

Subcommands:
  train              Train any model (--model, --seed/--seeds, --eval, ...)
  generate_splits    Build 70/15/15 split + fixed pair JSON files
  attention_profile  Compute class-conditional attention weights (Table 12)
  loeo               Leave-One-Event-Out generalisation experiment (Table 11)

python multisense.py train --help
python multisense.py loeo  --help
```

---

## Models

| Model | Role | Params | FLOPs | Memory |
|---|---|---|---|---|
| `ms_fusion` | **Proposed** — Siamese SqueezeNet-1.1 + channel-attention | 1.57 M | 700 M | 6.3 MB |
| `squeezenet_concat` | Fair dual-modality baseline (concat, no attention) | 1.25 M | 700 M | 5.0 MB |
| `lcnn` | Lightweight CNN single-modality baseline | 1.98 M | 1.37 B | 7.9 MB |
| `vit` | ViT-B/16 single-modality baseline | 86.57 M | 17.58 G | 346.3 MB |
| `vgg16` | VGG-16 single-modality baseline | 138.36 M | 15.47 G | 553.4 MB |

### MS Fusion Architecture (Equations 4–9)

```
f_sat = SqueezeNet(r_sat),  f_uav = SqueezeNet(r_uav)   # Siamese, D=512

f_c    = f_sat + f_uav                     (Eq. 4 — element-wise add)
f̃      = ReLU(BN(W_v · f_c))   W_v  ∈ R^{256×512}       (Eq. 5 — projection)
alpha  = σ(W_attn · f_c)       W_attn ∈ R^{256×512}       (Eq. 6 — attention mask)
m      = alpha ⊙ f̃                                        (Eq. 7 — masked features)
h      = ReLU(BN(W_r · m))     W_r  ∈ R^{256×256}         (Eq. 8 — refinement)
y      = W_cls · h              W_cls ∈ R^{5×256}           (Eq. 9 — classifier)
```

Weights: Xavier uniform. Biases: zero (Section 6.2).

---

## Dataset

### Five Disaster Classes

| Class | Event | Region | Label |
|---|---|---|---|
| `derna_flood` | Derna dam collapse, 2023 | Libya | 0 |
| `gaza_conflict` | Gaza conflict, 2023 | Palestine | 1 |
| `syria_turkey_eq` | Syria–Turkey earthquake, 2023 | Turkey/Syria | 2 |
| `hurricane_harvey` | Hurricane Harvey, 2017 | USA | 3 |
| `no_damage` | Pre-event control imagery | All four sites | 4 |

### Download

| Repository | Link |
|---|---|
| IEEE DataPort | https://doi.org/10.21227/cxy4-1136 |
| Mendeley Data | https://data.mendeley.com/datasets/krkft96n43/1 |
| GitHub Releases | https://github.com/Marwen200/Multisense/releases |

PlanetScope imagery: [Planet ERP](https://www.planet.com/markets/education-and-research/).

### Expected Data Layout

```
data/
  satellite/
    derna_flood/         ← satellite tiles
    gaza_conflict/
    syria_turkey_eq/
    hurricane_harvey/
    no_damage/
  uav/
    derna_flood/         ← UAV keyframes
    gaza_conflict/
    syria_turkey_eq/
    hurricane_harvey/
    no_damage/
  splits/                ← generated by generate_splits subcommand
    imbalanced_test_pairs.json
    balanced_test_pairs.json
    pairs_all.csv
    train_val_test_split.json
```

---

## Results

### Table 4 — Overall Performance (5-seed mean ± std)

| Model | Accuracy (%) | Macro-F1 (%) |
|---|---|---|
| L-CNN + ES (balanced) | 94.55 ± 0.18 | 94.38 ± 0.20 |
| L-CNN + ES (imbalanced) | 96.53 ± 0.15 | 95.87 ± 0.17 |
| ViT + ES | 98.53 ± 0.09 | 98.21 ± 0.10 |
| VGG16 + ES | 98.43 ± 0.11 | 98.06 ± 0.12 |
| SqueezeNet-Concat + ES | 98.87 ± 0.08 | 98.81 ± 0.09 |
| **MS Fusion + ES** | **99.57 ± 0.06** | **99.58 ± 0.06** |

### Table 11 — LOEO Generalisation (MS Fusion)

| Held-out Event | Tiles | Acc (%) | Macro-F1 (%) |
|---|---|---|---|
| Derna Flood | 600 | 87.83 | 86.14 |
| Gaza Conflict | 872 | 81.42 | 79.37 |
| Syria–Turkey EQ | 697 | 83.79 | 82.05 |
| Hurricane Harvey | 920 | 90.11 | 88.76 |
| **LOEO Mean** | — | **85.79** | **84.08** |

LOEO mean of 85.79% >> 20% random baseline → model learns damage features, not site fingerprints.

### Table 12 — Attention Profiles (representative run, seed 3)

| Class | Amplified (%) | Suppressed (%) | Neutral (%) |
|---|---|---|---|
| Derna Flood | 31.0 | 22.0 | 47.0 |
| Gaza Conflict | 27.5 | 25.5 | 47.0 |
| Syria–Turkey EQ | 24.5 | 28.0 | 47.5 |
| Hurricane Harvey | 29.0 | 21.5 | 49.5 |
| **No Damage** | **16.0** | **48.0** | 36.0 |

No Damage: highest suppression (48.0%) ↔ 4.28 pp ablation drop when mask removed.

---

## Hardware & Software

| | Specification |
|---|---|
| GPU | NVIDIA RTX 3050Ti |
| CPU | AMD Ryzen 7-4800H |
| RAM | 32 GB |
| PyTorch | 2.0.1 |
| CUDA | 11.8 |
| Python | 3.10 |
| torchvision | 0.15.2 |

---

## Citation

If you use this code or the MultiSense dataset, please cite:

```bibtex
@article{bouabid2025multisense,
  author    = {Bouabid, Marwen and Farah, Mohamed},
  title     = {{MultiSense}: A Knowledge-Guided Multimodal Benchmark and
               Attention Fusion Framework for Multi-Disaster Detection
               in Smart Cities},
  journal   = {The Visual Computer},
  publisher = {Springer},
  year      = {2025},
  url       = {https://github.com/Marwen200/Multisense},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```

> **GitHub citation note:** The BibTeX above references *The Visual Computer* as
> requested by the journal editors. It is also provided in `CITATION.cff` so
> GitHub's "Cite this repository" button returns the correct journal entry.

---

## License

MIT License — see [LICENSE](LICENSE).  
Dataset imagery subject to [Planet ERP](https://www.planet.com/markets/education-and-research/)
and [YouTube Data API v3](https://developers.google.com/youtube/terms/api-services-tos) terms.

---

**Contact:** Marwen Bouabid — `marwen.bouabid@ensi-uma.tn`  
ENSI, RIADI Lab LR99ES26, University of Manouba, 2010 Tunis, Tunisia
