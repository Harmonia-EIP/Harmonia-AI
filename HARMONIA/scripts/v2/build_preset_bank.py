#!/usr/bin/env python3
"""Build the v2 preset bank: sample presets, render them, embed the audio with CLAP.

Each preset is played at two notes (A2 and A4) with the app engine and both renders are
embedded with the CLAP teacher. Work is saved per chunk so an interrupted run resumes.

    python scripts/v2/build_preset_bank.py --n 200000 --out data/v2/bank
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.synth.engine import ENGINE_APP_1_1, SAMPLE_RATE, render_batch_physical  # noqa: E402
from src.synth.params import to_physical_batch  # noqa: E402
from src.synth.sampling import SOURCES, sample_bank  # noqa: E402

NOTES = np.array([45, 69])  # A2 (110 Hz) and A4 (440 Hz)
VELOCITY = 100 / 127
HOLD_S, TOTAL_S = 1.5, 4.0
SILENCE_RMS = 1e-4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--out", type=Path, default=BASE_DIR / "data" / "v2" / "bank")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--engine", type=int, default=ENGINE_APP_1_1)
    args = parser.parse_args()

    from src.v2.clap_audio import ClapEmbedder, TEACHER_ID, TEACHER_REVISION

    args.out.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.out / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    params, sources, labels = sample_bank(args.n, seed=args.seed)
    embedder = ClapEmbedder(half=True)
    n_chunks = (args.n + args.chunk - 1) // args.chunk
    start = time.time()
    for c in range(n_chunks):
        path = chunks_dir / f"{c:05d}.npz"
        if path.exists():
            continue
        lo, hi = c * args.chunk, min(args.n, (c + 1) * args.chunk)
        physical = to_physical_batch(params[lo:hi])
        audio = render_batch_physical(physical, NOTES, VELOCITY, HOLD_S, TOTAL_S, SAMPLE_RATE, args.engine, args.seed + lo * len(NOTES))
        flat = audio.reshape(-1, audio.shape[-1])
        rms = np.sqrt((np.nan_to_num(flat, nan=0.0).astype(np.float64) ** 2).mean(axis=1))
        finite = np.isfinite(flat).all(axis=1)
        emb = embedder.embed_audio([np.nan_to_num(w) for w in flat], batch_size=args.batch)
        np.savez(
            path,
            emb=emb.reshape(hi - lo, len(NOTES), -1).astype(np.float16),
            rms=rms.reshape(hi - lo, len(NOTES)).astype(np.float32),
            finite=finite.reshape(hi - lo, len(NOTES)),
        )
        done = c + 1
        rate = (hi) / max(1e-6, time.time() - start)
        print(f"chunk {done}/{n_chunks} | {hi} presets | {rate:.0f} presets/s | eta {(args.n - hi) / max(rate, 1e-6) / 60:.1f} min", flush=True)

    parts = [np.load(chunks_dir / f"{c:05d}.npz") for c in range(n_chunks)]
    emb = np.concatenate([p["emb"] for p in parts])
    rms = np.concatenate([p["rms"] for p in parts])
    finite = np.concatenate([p["finite"] for p in parts])
    keep = finite.all(axis=1) & (rms.max(axis=1) > SILENCE_RMS)
    np.savez(
        args.out / "bank.npz",
        params=params[keep],
        emb=emb[keep],
        rms=rms[keep],
        source=sources[keep],
        label=np.array(labels, dtype=object)[keep].astype(str),
        notes=NOTES,
    )
    meta = {
        "n_sampled": int(args.n),
        "n_kept": int(keep.sum()),
        "dropped_silent_or_nan": int((~keep).sum()),
        "notes": NOTES.tolist(),
        "velocity": VELOCITY,
        "hold_seconds": HOLD_S,
        "total_seconds": TOTAL_S,
        "sample_rate": SAMPLE_RATE,
        "engine": int(args.engine),
        "seed": args.seed,
        "sources": {s: int((sources[keep] == i).sum()) for i, s in enumerate(SOURCES)},
        "teacher": {"id": TEACHER_ID, "revision": TEACHER_REVISION},
    }
    (args.out / "bank.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
