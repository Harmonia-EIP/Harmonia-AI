#!/usr/bin/env python3
"""Build the v3 bank: real presets only, rendered and embedded with CLAP.

Entries: every DX7 voice (played by the msfa core, mode "dx7") and every OB-Xf / Surge preset whose
Harmonia version matched its original closely enough (mode "analog", 45 v3 parameters). Each one is
played at A2 and A4 like the v2 bank, and both renders are embedded with the CLAP teacher.

    python scripts/v3/build_bank_v3.py --max-distance 8
    -> data/v3/bank/bank.npz (emb, rms, kind, analog params, dx7 patches) + entries.jsonl (names, labels)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.presets import dx7_render  # noqa: E402
from src.synth import v3_params as P  # noqa: E402
from src.synth.engine_v3 import SAMPLE_RATE, render_batch  # noqa: E402

PRESET_DIR = BASE_DIR / "data" / "v3" / "presets"
MATCHED_DIR = BASE_DIR / "data" / "v3" / "matched"
NOTES = np.array([45, 69])  # A2, A4 (as in the v2 bank)
VELOCITY = 100
HOLD_S, TOTAL_S = 1.5, 4.0
SILENCE_RMS = 1e-4


def load_entries(max_distance: float):
    entries = []
    for line in open(PRESET_DIR / "dx7.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        entries.append({"id": rec["id"], "kind": "dx7", "source": "dx7", "name": rec["name"],
                        "aliases": rec["aliases"], "category": "", "author": rec["author"],
                        "cartridge": rec.get("cartridge", ""), "comment": "", "voice": rec["params"]})
    for source in ("obxf", "surge"):
        path = MATCHED_DIR / f"{source}.jsonl"
        if not path.exists():
            continue
        meta = {}
        for line in open(PRESET_DIR / f"{source}.jsonl", encoding="utf-8"):
            rec = json.loads(line)
            meta[rec["id"]] = rec
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            if row.get("status") != "ok" or row["distance"] > max_distance:
                continue
            rec = meta[row["id"]]
            entries.append({"id": row["id"], "kind": "analog", "source": source, "name": rec["name"],
                            "aliases": rec["aliases"], "category": rec["category"], "author": rec["author"],
                            "cartridge": "", "comment": rec.get("comment", ""), "distance": row["distance"],
                            "physical": row["physical"]})
    return entries


def render(entry) -> np.ndarray:
    if entry["kind"] == "dx7":
        return np.stack([dx7_render.render(entry["voice"], note=int(n), velocity=VELOCITY, hold_seconds=HOLD_S,
                                           total_seconds=TOTAL_S, sample_rate=SAMPLE_RATE) for n in NOTES])
    p = np.asarray(entry["physical"], dtype=np.float64)[None, :]
    return render_batch(p, NOTES, VELOCITY / 127, HOLD_S, TOTAL_S, SAMPLE_RATE, 7)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-distance", type=float, default=8.0)
    parser.add_argument("--out", type=Path, default=BASE_DIR / "data" / "v3" / "bank")
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--only", choices=["dx7", "analog"], help="embed one kind only (no bank file)")
    args = parser.parse_args()

    from src.v2.clap_audio import TEACHER_ID, TEACHER_REVISION, ClapEmbedder

    entries = load_entries(args.max_distance)
    if args.only:
        entries = [e for e in entries if e["kind"] == args.only]
    args.out.mkdir(parents=True, exist_ok=True)
    chunks = args.out / "chunks"
    chunks.mkdir(exist_ok=True)
    embedder = ClapEmbedder(half=True)
    start = time.time()
    names = []
    # DX7 voices never change: cache them by position. Analog chunks are keyed by their content so a new
    # matching run (other presets, other parameters) never reuses stale embeddings.
    for kind in ("dx7", "analog"):
        group = [e for e in entries if e["kind"] == kind]
        for c in range((len(group) + args.chunk - 1) // args.chunk):
            part = group[c * args.chunk:(c + 1) * args.chunk]
            key = f"{c:05d}" if kind == "dx7" else hashlib.sha1(json.dumps(
                [(e["id"], e["physical"]) for e in part]).encode(), usedforsecurity=False).hexdigest()[:12]
            path = chunks / f"{kind}_{key}.npz"
            names.append(path)
            if path.exists():
                continue
            audio = np.stack([render(e) for e in part]).astype(np.float32)
            flat = np.nan_to_num(audio.reshape(-1, audio.shape[-1]))
            rms = np.sqrt((flat.astype(np.float64) ** 2).mean(axis=1))
            emb = embedder.embed_audio(list(flat), batch_size=args.batch)
            np.savez(path, emb=emb.reshape(len(part), len(NOTES), -1).astype(np.float16),
                     rms=rms.reshape(len(part), len(NOTES)).astype(np.float32))
            print(f"{kind} chunk {c + 1} | {min(len(group), (c + 1) * args.chunk)}/{len(group)} | "
                  f"{time.time() - start:.0f}s", flush=True)
    entries = [e for kind in ("dx7", "analog") for e in entries if e["kind"] == kind]
    if args.only:
        return 0

    parts = [np.load(path) for path in names]
    emb = np.concatenate([p["emb"] for p in parts])
    rms = np.concatenate([p["rms"] for p in parts])
    keep = rms.max(axis=1) > SILENCE_RMS
    kept = [e for e, k in zip(entries, keep) if k]
    analog = np.array([e.get("physical", P.DEFAULTS.tolist()) for e in kept], dtype=np.float32)
    patches = np.array([list(dx7_render.to_patch(e["voice"])) if e["kind"] == "dx7" else [0] * 156 for e in kept],
                       dtype=np.uint8)
    np.savez(args.out / "bank.npz", emb=emb[keep], rms=rms[keep], kind=np.array([e["kind"] for e in kept]),
             analog=analog, dx7=patches, notes=NOTES)
    with open(args.out / "entries.jsonl", "w", encoding="utf-8") as f:
        for e in kept:
            f.write(json.dumps({k: v for k, v in e.items() if k not in ("voice", "physical")}, ensure_ascii=False) + "\n")
    meta = {"entries": len(kept), "dropped_silent": int((~keep).sum()),
            "by_source": {s: sum(e["source"] == s for e in kept) for s in ("dx7", "obxf", "surge")},
            "max_distance": args.max_distance, "notes": NOTES.tolist(), "hold_seconds": HOLD_S,
            "total_seconds": TOTAL_S, "teacher": {"id": TEACHER_ID, "revision": TEACHER_REVISION}}
    (args.out / "bank.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
