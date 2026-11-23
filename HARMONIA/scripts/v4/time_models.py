#!/usr/bin/env python3
"""Seconds per generation of several model folders on the same few cores (#35).

The VPS has 2 cores: every thread pool (torch, numba, BLAS) is limited to --threads before anything is
imported, each model is warmed up, then each generates the same prompts with the server's settings (no time
budget, so both do the same work). v3 and v4 share the run time and their networks have the same sizes, so
the times should match; this checks it.

    python scripts/v4/time_models.py models/harmonia_v3 models/harmonia_v4 --threads 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
PROMPTS = ["Porte qui claque", "Grosse basse ronflante", "Verre qui se casse", "Acid pluck lead",
           "Soft pad atmospheric", "Piano in a big room", "Pluie sur un toit en tôle", "thunder", "footsteps on gravel",
           "Bright brass stab", "Cloche lointaine", "Kick 808 profond"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dirs", nargs="+", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=2, help="passes over the prompts (variations 0, 1...)")
    args = parser.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    if str(BASE_DIR) not in sys.path:
        sys.path.append(str(BASE_DIR))

    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    from src.v3.inference import HarmoniaV3

    report = {}
    for model_dir in args.model_dirs:
        model = HarmoniaV3(model_dir)
        per_sound = model.warm_up()
        seconds, heard = [], []
        for variation in range(args.repeat):
            for prompt in PROMPTS:
                t = time.perf_counter()
                result = model.generate(prompt, variation=variation)
                seconds.append(time.perf_counter() - t)
                heard.append(result.info.get("heard", 0))
        report[str(model_dir)] = {"ms_per_sound": round(1000 * per_sound, 1),
                                  "seconds_mean": round(float(np.mean(seconds)), 2),
                                  "seconds_p90": round(float(np.percentile(seconds, 90)), 2),
                                  "sounds_per_prompt": round(float(np.mean(heard)), 1)}
        print(model_dir, json.dumps(report[str(model_dir)]), flush=True)
        del model
    return 0


if __name__ == "__main__":
    sys.exit(main())
