#!/usr/bin/env python3
"""Embed the FSD50K recordings with the CLAP teacher (audio side).

Writes data/v2/fsd50k_<split>.npz with fname, labels (leaf first), title and the (N, 512) embeddings.

    python scripts/v2/embed_fsd50k.py --fsd50k ~/Datasets/FSD50K --split dev
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.v2.clap_audio import CLAP_SR, MAX_SAMPLES, fit_to_window  # noqa: E402


def load_clip(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != CLAP_SR:
        audio = soxr.resample(audio, sr, CLAP_SR, quality="HQ").astype(np.float32)
    return fit_to_window(audio, MAX_SAMPLES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd50k", type=Path, default=Path.home() / "Datasets" / "FSD50K")
    parser.add_argument("--split", choices=("dev", "eval"), default="dev")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    from src.v2.clap_audio import ClapEmbedder

    out = args.out or BASE_DIR / "data" / "v2" / f"fsd50k_{args.split}.npz"
    with open(args.fsd50k / "FSD50K.ground_truth" / f"{args.split}.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    info = json.loads((args.fsd50k / "FSD50K.metadata" / f"{args.split}_clips_info_FSD50K.json").read_text(encoding="utf-8"))
    audio_dir = args.fsd50k / f"FSD50K.{args.split}_audio"

    embedder = ClapEmbedder(half=True)
    embeddings = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i in range(0, len(rows), args.batch):
            chunk = rows[i : i + args.batch]
            waves = list(pool.map(load_clip, [audio_dir / f"{r['fname']}.wav" for r in chunk]))
            embeddings.append(embedder.embed_audio(waves, batch_size=args.batch).astype(np.float16))
            if (i // args.batch) % 50 == 0:
                done = i + len(chunk)
                rate = done / max(1e-6, time.time() - start)
                print(f"{done}/{len(rows)} | {rate:.0f} clips/s | eta {(len(rows) - done) / max(rate, 1e-6) / 60:.1f} min", flush=True)

    np.savez(
        out,
        fname=np.array([r["fname"] for r in rows]),
        labels=np.array([r["labels"] for r in rows]),
        split=np.array([r.get("split", args.split) for r in rows]),
        title=np.array([info.get(r["fname"], {}).get("title", "") for r in rows]),
        emb=np.concatenate(embeddings),
    )
    print(f"wrote {out} ({len(rows)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
