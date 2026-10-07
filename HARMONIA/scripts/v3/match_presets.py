#!/usr/bin/env python3
"""Convert real presets to Harmonia v3 and refine them against the original synthesizer.

For each preset: play it in its own synth (pedalboard) at C3/C4/C5, convert its parameters
(src/presets/convert_*.py), then refine the continuous parameters by sound matching (src/presets/matching.py).
Writes one JSON line per preset; a sample of presets also keeps original/Harmonia audio for listening.

    python scripts/v3/match_presets.py --source obxf --workers 10
"""

from __future__ import annotations

import os

# One thread per worker: numpy/Accelerate and numba would otherwise each start a thread per core in
# every worker process and the machine thrashes (load > 250 on 14 cores). Set before numpy is imported;
# spawned workers inherit it.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.v3.fetch_presets import DEFAULT_OUT  # noqa: E402

PRESET_DIR = BASE_DIR / "data" / "v3" / "presets"
OUT_DIR = BASE_DIR / "data" / "v3" / "matched"
_state: dict = {}


def _converter(source: str):
    if source == "obxf":
        from src.presets.convert_obxf import obxf_to_v3
        return lambda rec: obxf_to_v3(rec["params"])
    from src.presets.convert_surge import surge_to_v3
    return lambda rec: surge_to_v3(rec["params"], rec.get("modulation", []))


def _init(source: str, budget: int, audio_dir: str) -> None:
    from src.presets.originals import OriginalSynth

    _state.update(synth=OriginalSynth(source), convert=_converter(source), source=source, budget=budget,
                  root=DEFAULT_OUT / source, audio_dir=Path(audio_dir))


def _work(task):
    from src.presets import matching as M

    rec, keep_audio = task
    t0 = time.time()
    synth = _state["synth"]
    try:
        synth.load(_state["root"] / rec["paths"][0])
        original = [synth.render(note=n, velocity=M.VELOCITY, hold_seconds=M.HOLD_SECONDS,
                                 total_seconds=M.TOTAL_SECONDS) for n in M.NOTES]
        loudness = float(np.sqrt(np.mean([np.mean(a ** 2) for a in original])))
        if not np.isfinite(loudness) or loudness < 1e-5:
            return {"id": rec["id"], "status": "silent"}
        target = M.features(original)
        start = _state["convert"](rec)
        result = M.match(start, target, budget=_state["budget"], seed=int(rec["id"].split(":")[1], 16) % 2**32,
                         search_discrete=_state["source"] == "surge")
    except Exception as exc:  # a broken preset must not stop the batch
        return {"id": rec["id"], "status": f"error: {exc}"}
    if keep_audio:
        harmonia = M.render_notes(result.physical)
        np.savez_compressed(_state["audio_dir"] / f"{rec['id'].replace(':', '_')}.npz",
                            original=np.stack(original).astype(np.float32),
                            harmonia=np.stack(harmonia).astype(np.float32),
                            converted=np.stack(M.render_notes(start)).astype(np.float32))
    return {"id": rec["id"], "status": "ok", "name": rec["name"], "category": rec["category"],
            "start_distance": round(result.start_distance, 4), "distance": round(result.distance, 4),
            "physical": [round(float(x), 6) for x in result.physical], "seconds": round(time.time() - t0, 1),
            "audio": keep_audio}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["obxf", "surge"], required=True)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 4))
    parser.add_argument("--budget", type=int, default=240)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bank", default=None, help="Surge only: factory or 3rdparty")
    parser.add_argument("--listen", type=int, default=60, help="presets that keep their audio for listening")
    args = parser.parse_args()

    records = [json.loads(line) for line in open(PRESET_DIR / f"{args.source}.jsonl", encoding="utf-8")]
    from src.presets.filters import playable

    records = [r for r in records if playable(r)]
    if args.bank:
        records = [r for r in records if r.get("bank") == args.bank]
    if args.limit:
        records = records[:args.limit]
    out = OUT_DIR / f"{args.source}.jsonl"
    audio_dir = OUT_DIR / f"{args.source}_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in open(out, encoding="utf-8")}
    rng = np.random.default_rng(0)
    listen = set(rng.choice(len(records), size=min(args.listen, len(records)), replace=False).tolist())
    tasks = [(rec, i in listen) for i, rec in enumerate(records) if rec["id"] not in done]
    print(f"{args.source}: {len(tasks)} presets to match ({len(done)} already done), {args.workers} workers")
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(args.workers, initializer=_init, initargs=(args.source, args.budget, str(audio_dir))) as pool, \
            open(out, "a", encoding="utf-8") as f:
        for k, row in enumerate(pool.imap_unordered(_work, tasks), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if k % 25 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
