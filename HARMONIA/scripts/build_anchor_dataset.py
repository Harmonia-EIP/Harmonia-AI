#!/usr/bin/env python3
"""Build a charter-compatible dataset from `data/raw/anchor_presets.json`
with massive prompt-level oversampling so anchors actually dominate training.

Why: the production charter dataset (52k Sylenth1 dumps) drowns the 47 anchor
records at a ratio of ~0.09%. Train on this output instead (or mix it in) to
let hand-crafted concepts steer the model.

For each `{label, prompts, parameters}` block we emit one record per prompt
per replication, with `prompt_augment` in the dataloader providing further
phrasing variance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.prepare_dataset import _iter_anchor_records

DEFAULT_INPUT = BASE_DIR / "data" / "raw" / "anchor_presets.json"
DEFAULT_OUTPUT = BASE_DIR / "data" / "processed" / "anchor_oversample.charter.npy"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicate", type=int, default=200, help="How many copies of each anchor prompt to write.")
    parser.add_argument("--mix", type=Path, default=None, help="Optional charter NPY to concatenate (e.g. cleaned_dataset.charter.clean.npy).")
    args = parser.parse_args()

    base_records = list(_iter_anchor_records(args.input))
    if not base_records:
        print(f"No anchor records loaded from {args.input}")
        return 1

    records = []
    for _ in range(args.replicate):
        records.extend(base_records)

    extra = 0
    if args.mix is not None and args.mix.exists():
        loaded = np.load(args.mix, allow_pickle=True).tolist()
        extra = len(loaded)
        records.extend(loaded)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.array(records, dtype=object), allow_pickle=True)

    unique_labels = {str(r.get("raw_text", "")) for r in base_records}
    print(f"Anchor prompts (raw): {len(base_records)}  (across {len(unique_labels)} unique texts)")
    print(f"Replication factor  : {args.replicate}")
    print(f"Mixed-in records    : {extra}")
    print(f"Total written       : {len(records)}")
    print(f"Output              : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
