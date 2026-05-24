#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# train_all.sh — Reproduce ALL MultiSense results from the paper.
#
# Calls multisense.py (the single implementation file) for every step:
#   1. generate_splits   — build fixed pair files (run once)
#   2. train lcnn        — L-CNN balanced + imbalanced baselines
#   3. train vit         — ViT-B/16 baseline
#   4. train vgg16       — VGG-16 baseline
#   5. train squeezenet_concat — fair dual-modality baseline
#   6. train ms_fusion   — proposed model (5 seeds)
#   7. attention_profile — Table 12 (seed 3 representative run)
#   8. loeo              — Table 11 (leave-one-event-out)
#
# Wall-clock estimates on RTX 3050Ti (Table 14 of the paper):
#   ms_fusion         ~3 h 37 m (56 epochs × 3 m 53 s)
#   squeezenet_concat ~3 h 26 m
#   lcnn imbalanced   ~6 h 32 m
#   vit               ~15 h 31 m
#   vgg16             ~34 h 35 m
#
# Usage:
#   bash train_all.sh
#   DATA_ROOT=/path/to/data SEEDS=1 bash train_all.sh   # quick smoke test
#
# Cite:
#   Bouabid, M., & Farah, M. (2025). MultiSense. The Visual Computer.
#   https://github.com/Marwen200/Multisense
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
SPLITS_DIR="${DATA_ROOT}/splits"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
SEEDS="${SEEDS:-5}"

PY="python multisense.py"
TS() { date '+%Y-%m-%d %H:%M:%S'; }

echo "════════════════════════════════════════════════════════════════"
echo "  MultiSense — Full Reproduction Run"
echo "  multisense.py version (single-file)"
echo "  Data : ${DATA_ROOT}   Output: ${OUTPUT_DIR}"
echo "  Seeds: 0-$((SEEDS-1))   Device: ${DEVICE}"
echo "  Start: $(TS)"
echo "════════════════════════════════════════════════════════════════"

# ── Step 1: Generate fixed splits (run once) ──────────────────────────────────
echo ""
echo "[$(TS)]  Step 1 — generate_splits"
${PY} generate_splits \
    --data_root  "${DATA_ROOT}" \
    --output_dir "${SPLITS_DIR}" \
    --seed 42

PAIRS_CSV="${SPLITS_DIR}/pairs_all.csv"

# Helper: train one model over SEEDS seeds
run_model() {
    local MODEL=$1; shift
    local EXTRA="${*:-}"
    echo ""
    echo "[$(TS)]  ── ${MODEL} ──"
    ${PY} train \
        --model       "${MODEL}" \
        --seeds       "${SEEDS}" \
        --data_root   "${DATA_ROOT}" \
        --output_dir  "${OUTPUT_DIR}" \
        --pairs_csv   "${PAIRS_CSV}" \
        --max_epochs  "${MAX_EPOCHS}" \
        --device      "${DEVICE}" \
        --num_workers "${WORKERS}" \
        ${EXTRA}
    echo "[$(TS)]  ${MODEL} done"
}

# ── Step 2: L-CNN baselines ───────────────────────────────────────────────────
echo ""
echo "[$(TS)]  Step 2 — L-CNN balanced (Table 4, rows 1-2)"
run_model lcnn "--balanced_train --balanced_test"

echo ""
echo "[$(TS)]  Step 3 — L-CNN imbalanced (Table 4, row 3)"
run_model lcnn

# ── Step 3: Heavy single-modality baselines ───────────────────────────────────
echo ""
echo "[$(TS)]  Step 4 — ViT-B/16 (Table 4, row 4)"
run_model vit

echo ""
echo "[$(TS)]  Step 5 — VGG-16 (Table 4, row 5)"
run_model vgg16

# ── Step 4: Dual-modality models ──────────────────────────────────────────────
echo ""
echo "[$(TS)]  Step 6 — SqueezeNet-Concat (Table 4, row 6)"
run_model squeezenet_concat

echo ""
echo "[$(TS)]  Step 7 — MS Fusion / proposed model (Table 4, row 7)"
run_model ms_fusion

# ── Step 5: Attention profile (Table 12) ─────────────────────────────────────
# Seed 3 is the representative run (closest to 5-seed mean)
BEST_CKPT="${OUTPUT_DIR}/ms_fusion/seed_3/best.pt"
echo ""
echo "[$(TS)]  Step 8 — Attention profile (Table 12, seed 3)"
if [ -f "${BEST_CKPT}" ]; then
    ${PY} attention_profile \
        --checkpoint  "${BEST_CKPT}" \
        --data_root   "${DATA_ROOT}" \
        --pairs_csv   "${PAIRS_CSV}" \
        --output_dir  "${OUTPUT_DIR}/ms_fusion/seed_3" \
        --device      "${DEVICE}" \
        --num_workers "${WORKERS}"
else
    echo "  [WARN] ${BEST_CKPT} not found — skipping attention profile"
fi

# ── Step 6: LOEO experiment (Table 11) ────────────────────────────────────────
echo ""
echo "[$(TS)]  Step 9 — LOEO experiment (Table 11)"
${PY} loeo \
    --data_root   "${DATA_ROOT}" \
    --output_dir  "${OUTPUT_DIR}/loeo" \
    --pairs_csv   "${PAIRS_CSV}" \
    --max_epochs  "${MAX_EPOCHS}" \
    --device      "${DEVICE}" \
    --num_workers "${WORKERS}"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All done.  $(TS)"
echo "  Results in: ${OUTPUT_DIR}"
echo "════════════════════════════════════════════════════════════════"
