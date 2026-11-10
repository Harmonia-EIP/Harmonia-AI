#!/usr/bin/env python3
"""Benchmark of the v3 run time, step 1: generate a preset per prompt and write its sound.

Prompts:
- Malo's (benchmarks/v3_prompts_p1.json and v3_prompts_p2.json, never used for training);
- the class names of FSD50K's eval split, as their annotators wrote them ("Slam", "Walk and footsteps"...),
  human voice classes left out: each has real recordings to compare the generated sound with.

One system per run (a name and its settings); scripts/v3/benchmark_judge_v3.py then scores the folders with
an independent judge.

    python scripts/v3/benchmark_v3.py --name v3.0 --polish-rounds 0
    python scripts/v3/benchmark_v3.py --name polish2 --polish-rounds 2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.v3.match_recordings import VOICE  # noqa: E402
from src import charter_v3  # noqa: E402
from src.synth import v3_params as P  # noqa: E402

OUT_DIR = BASE_DIR / "data" / "v3" / "bench"
NOTE, HOLD_SECONDS, TOTAL_SECONDS, SR = 57, 1.5, 4.0, 48000  # A3: not one of the notes the run time listens to


def prompts(fsd50k: Path, limit_classes: int = 0):
    rows = [{"set": "malo", "prompt": p, "judge": p} for p in
            json.loads((BASE_DIR / "benchmarks" / "v3_prompts_p1.json").read_text(encoding="utf-8"))]
    rows += [{"set": "malo", **p} for p in
             json.loads((BASE_DIR / "benchmarks" / "v3_prompts_p2.json").read_text(encoding="utf-8"))]
    with open(fsd50k / "FSD50K.ground_truth" / "eval.csv", newline="", encoding="utf-8") as f:
        classes = sorted({r["labels"].split(",")[0] for r in csv.DictReader(f)} - VOICE)
    if limit_classes:
        classes = classes[::max(1, len(classes) // limit_classes)][:limit_classes]
    for name in classes:
        text = name.replace("_", " ").lower()
        rows.append({"set": "fsd", "prompt": text, "judge": text, "fsd_class": name})
    return rows


def play(preset: dict) -> np.ndarray:
    """Mono audio of a charter-v3 preset at A3."""
    engine, values = charter_v3.read_preset(preset)
    if engine == "dx7":
        from src.presets import dx7_render

        return dx7_render.render(charter_v3.dx7_voice(values), note=NOTE, hold_seconds=HOLD_SECONDS,
                                 total_seconds=TOTAL_SECONDS, sample_rate=SR)
    from src.synth.engine_v3 import render

    return render(P.physical_from_dict(values), NOTE, 100 / 127, HOLD_SECONDS, TOTAL_SECONDS, SR).mean(axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-dir", type=Path, default=BASE_DIR / "models" / "harmonia_v3")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--polish-rounds", type=int, default=2)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--fsd50k", type=Path, default=Path.home() / "Datasets" / "FSD50K")
    parser.add_argument("--classes", type=int, default=0, help="only this many FSD50K classes (0: all)")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    from src.v3.inference import HarmoniaV3, Settings

    model = HarmoniaV3(args.model_dir, settings=Settings(candidates=args.candidates, polish_rounds=args.polish_rounds,
                                                         population=args.population))
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i, row in enumerate(prompts(args.fsd50k, args.classes)):
        t = time.time()
        result = model.generate(row["prompt"])
        audio = np.nan_to_num(play(result.preset)).astype(np.float32)
        sf.write(out / f"{i:04d}.wav", audio, SR)
        rows.append({**row, "wav": f"{i:04d}.wav", "engine": result.engine, "similarity": result.similarity,
                     "seconds": round(time.time() - t, 2), "peak": float(np.abs(audio).max()),
                     "parameters": result.preset["parameters"]})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1} prompts ({time.time() - t0:.0f}s)", flush=True)
    (out / "generations.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    mean = float(np.mean([r["similarity"] for r in rows]))
    print(f"{args.name}: {len(rows)} prompts, mean similarity {mean:.3f}, "
          f"{np.mean([r['seconds'] for r in rows]):.1f}s per prompt, "
          f"dx7 {sum(r['engine'] == 'dx7' for r in rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
