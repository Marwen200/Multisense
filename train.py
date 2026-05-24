#!/usr/bin/env python3
"""
train.py — Convenience wrapper matching the response letter command exactly:

    python train.py --model ms_fusion --seed 3 --data_root ./data

Equivalent to:

    python multisense.py train --model ms_fusion --seed 3 --data_root ./data

All logic lives in multisense.py (single-file design). This file exists
only so the exact CLI promised in the response letter to The Visual Computer
works without modification.

Cite:
    Bouabid, M., & Farah, M. (2025). MultiSense: A Knowledge-Guided Multimodal
    Benchmark and Attention Fusion Framework for Multi-Disaster Detection in
    Smart Cities. The Visual Computer. https://github.com/Marwen200/Multisense
"""

import sys

# Insert the "train" subcommand before the user's arguments so that
# multisense.py's argparse dispatcher routes to cmd_train().
sys.argv = [sys.argv[0], "train"] + sys.argv[1:]

from multisense import main  # noqa: E402

if __name__ == "__main__":
    main()
