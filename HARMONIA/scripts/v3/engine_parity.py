#!/usr/bin/env python3
"""Does the plugin's C++ engine sound like the reference engine the v3 models learned? (Harmonia-EIP/Harmonia-App#49)

Builds Harmonia-APP's `src/engine/HarmoniaEngineV3.cpp` into a shared library, renders real presets of the v3
bank with both engines (C2, C3, C4, stereo) and compares them with the perceptual distance used to match presets
by ear (src/presets/perceptual.py). Both engines draw random unison phases, drift, noise and sample & hold, so
two renders never agree sample for sample: the yardstick is the distance between two renders of the reference
itself with different seeds. Also reported: the level difference (the distance normalizes loudness) and, for
presets without randomness that matters (one voice, no noise, no sample & hold), the waveform correlation.

    python scripts/v3/engine_parity.py --app ~/Harmonia-APP/.worktrees/ai-v3 --presets 80
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess  # nosec B404 - fixed compiler invocation
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.presets import perceptual  # noqa: E402
from src.synth import v3_params as P  # noqa: E402
from src.synth.engine_v3 import render  # noqa: E402

BANK = BASE_DIR / "data" / "v3" / "bank"
BUILD = BASE_DIR / "data" / "v3" / "build" / "engine"
OUT = BASE_DIR / "data" / "v3" / "engine_parity"
NOTES = (36, 48, 60)
VELOCITY, HOLD, TOTAL, SR = 100 / 127, 1.5, 4.0, 48000


def build(app: Path) -> Path:
    source = app / "src" / "engine" / "HarmoniaEngineV3.cpp"
    BUILD.mkdir(parents=True, exist_ok=True)
    library = BUILD / ("libharmonia_engine.dylib" if sys.platform == "darwin" else "libharmonia_engine.so")
    command = ["c++", "-std=c++17", "-O2", "-shared", "-fPIC", str(source), "-o", str(library)]
    subprocess.run(command, check=True)  # nosec B603 - fixed arguments
    return library


class CppEngine:
    def __init__(self, library: Path):
        self.lib = ctypes.CDLL(str(library))
        f = self.lib.harmonia_render_analog_note
        f.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                      ctypes.c_double, ctypes.c_int, ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        f.restype = ctypes.c_int

    def render(self, physical: np.ndarray, note: int, seed: int) -> np.ndarray:
        p = np.ascontiguousarray(physical, dtype=np.float64)
        n = int(TOTAL * SR)
        out = np.zeros(2 * n, dtype=np.float32)
        got = self.lib.harmonia_render_analog_note(p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(p), note,
                                                   VELOCITY, HOLD, TOTAL, SR, seed,
                                                   out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(out))
        if got != n:
            raise RuntimeError(f"C++ engine refused the render ({got})")
        return out.reshape(2, n)


def level_db(notes) -> float:
    return float(10 * np.log10(np.mean([np.mean(a.astype(np.float64) ** 2) for a in notes]) + 1e-20))


def deterministic(p: np.ndarray) -> bool:
    return p[P.INDEX["unison_voices"]] == 1 and p[P.INDEX["noise_level"]] == 0 and p[P.INDEX["lfo_waveform"]] != 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", type=Path, default=Path.home() / "Harmonia-APP")
    parser.add_argument("--presets", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    cpp = CppEngine(build(args.app))
    z = np.load(BANK / "bank.npz")
    entries = [json.loads(line) for line in open(BANK / "entries.jsonl", encoding="utf-8")]
    analog = np.flatnonzero(z["kind"] == "analog")
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(analog, min(args.presets, len(analog)), replace=False)

    rows = []
    for k, i in enumerate(chosen):
        p = z["analog"][i].astype(np.float64)
        ref_a = [render(p, n, VELOCITY, HOLD, TOTAL, SR, 1000 + 7 * j) for j, n in enumerate(NOTES)]
        ref_b = [render(p, n, VELOCITY, HOLD, TOTAL, SR, 2000 + 7 * j) for j, n in enumerate(NOTES)]
        plug = [cpp.render(p, n, 3000 + 7 * j) for j, n in enumerate(NOTES)]
        da, db, dc = perceptual.describe(ref_a), perceptual.describe(ref_b), perceptual.describe(plug)
        row = {"id": entries[i]["id"], "name": entries[i]["name"],
               "reference_vs_reference": perceptual.distance(da, db),
               "reference_vs_plugin": perceptual.distance(da, dc),
               "plugin_parts": perceptual.components(da, dc),
               "level_db": level_db(plug) - level_db(ref_a)}
        if deterministic(p):
            a, b = ref_a[1][:, : SR // 10].ravel(), plug[1][:, : SR // 10].ravel()
            row["correlation_100ms"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
        rows.append(row)
        print(f"{k + 1:3d} {row['name'][:24]:24s} ref/ref {row['reference_vs_reference']:5.2f}  "
              f"ref/plugin {row['reference_vs_plugin']:5.2f}  level {row['level_db']:+5.2f} dB"
              + (f"  corr {row['correlation_100ms']:.4f}" if row.get("correlation_100ms") is not None else ""), flush=True)

    rr = np.array([r["reference_vs_reference"] for r in rows])
    rp = np.array([r["reference_vs_plugin"] for r in rows])
    levels = np.array([r["level_db"] for r in rows])
    corr = [r["correlation_100ms"] for r in rows if r.get("correlation_100ms") is not None]
    summary = {
        "presets": len(rows),
        "reference_vs_reference_median": float(np.median(rr)),
        "reference_vs_plugin_median": float(np.median(rp)),
        "plugin_within_reference_spread": float(np.mean(rp <= rr * 1.5 + 0.5)),
        "level_db_median": float(np.median(levels)), "level_db_max_abs": float(np.abs(levels).max()),
        "deterministic_presets": len(corr), "correlation_100ms_min": float(min(corr)) if corr else None,
        "worst": sorted(rows, key=lambda r: r["reference_vs_reference"] - r["reference_vs_plugin"])[:8],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "worst"}, indent=1))
    print("worst:", [(w["name"], round(w["reference_vs_reference"], 2), round(w["reference_vs_plugin"], 2)) for w in summary["worst"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
