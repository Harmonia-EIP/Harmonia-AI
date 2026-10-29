#!/usr/bin/env python3
"""Per-preset retrieval priors for the bank, saved to data/v2/bank/bank_extra.npz:

- audible_db: level above 60 Hz (max over the two notes). Near-silent presets (rumble far below
  the audible range) embed close to everything in CLAP and must not be retrieved;
- hub: mean cosine to the 10 closest corpus sentences (CLAP teacher). Used for CSLS: presets
  that are close to every description ("hubs") get penalized at retrieval time.

    python scripts/v2/analyze_bank.py --data data/v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.v2.build_preset_bank import HOLD_S, NOTES, TOTAL_S, VELOCITY  # noqa: E402
from scripts.v2.train_predictor import load_bank  # noqa: E402
from src.synth.engine import ENGINE_APP_1_1, SAMPLE_RATE, audible_rms_batch_physical  # noqa: E402
from src.synth.params import to_physical_batch  # noqa: E402

HUB_NEIGHBOURS = 10


def hubness(bank_emb: np.ndarray, texts: np.ndarray, k: int = HUB_NEIGHBOURS, chunk: int = 1024) -> np.ndarray:
    out = np.empty(len(bank_emb), dtype=np.float32)
    for i in range(0, len(bank_emb), chunk):
        sims = bank_emb[i : i + chunk] @ texts.T
        top = np.partition(sims, -k, axis=1)[:, -k:]
        out[i : i + chunk] = top.mean(axis=1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "v2")
    parser.add_argument("--chunk", type=int, default=4096)
    args = parser.parse_args()

    bank_params, bank_emb = load_bank(args.data / "bank" / "bank.npz")
    levels = []
    for i in range(0, len(bank_params), args.chunk):
        physical = to_physical_batch(bank_params[i : i + args.chunk])
        rms = audible_rms_batch_physical(physical, NOTES, VELOCITY, HOLD_S, TOTAL_S, SAMPLE_RATE, ENGINE_APP_1_1, 60.0)
        levels.append(rms.max(axis=1))
        print(f"loudness {min(i + args.chunk, len(bank_params))}/{len(bank_params)}", flush=True)
    audible_db = 20.0 * np.log10(np.maximum(np.nan_to_num(np.concatenate(levels)), 1e-9))

    texts = np.load(args.data / "teacher_text.npz")["emb"].astype(np.float32)
    hub = hubness(bank_emb, texts)
    np.savez(args.data / "bank" / "bank_extra.npz", audible_db=audible_db.astype(np.float32), hub=hub)
    summary = {
        "presets": int(len(bank_params)),
        "audible_db_percentiles": np.percentile(audible_db, [1, 5, 25, 50]).round(1).tolist(),
        "below_-40dB": int((audible_db < -40).sum()),
        "hub_percentiles": np.percentile(hub, [1, 50, 99]).round(3).tolist(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
