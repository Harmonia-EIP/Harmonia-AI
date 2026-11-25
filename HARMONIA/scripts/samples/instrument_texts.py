#!/usr/bin/env python3
"""Pairs (name of a recorded instrument, sound of its recordings) for the text -> sound prior (#47).

The names are the libraries' own: the folders their authors filed each instrument in ("Strings / Solo Violin /
arco-vib", "Chordophones / Zithers / Grand Piano, Steinway B") and the catalogue's name built from them
("Solo Violin · Arco Vib"), "Salamander Grand Piano" for the piano. Each name is paired with CLAP's hearing of
several of the instrument's own recordings (full pack): a few notes across its range, at a middle and a strong
velocity. Nothing is written by hand.

    python scripts/samples/instrument_texts.py     # -> data/samples/instrument_pairs.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

ROOT = Path.home() / "Datasets" / "harmonia-samples"
OUT = BASE_DIR / "data" / "samples" / "instrument_pairs.npz"
NOTES, SECONDS, SR = 4, 4.0, 48000
GENERIC = {"samples", "sustains", "sustain"}  # folder names that say nothing about the instrument


def names(instrument: dict, original: dict) -> list:
    """The instrument's names: the catalogue's, and the authors' folder path."""
    texts = [instrument["name"].replace(" · ", ", ")]
    first = next(r for r in original["regions"] if r["trigger"] == "attack")
    folders = [p for p in PurePosixPath(first["file"]).parent.parts if p.lower() not in GENERIC]
    if folders:
        texts.append(", ".join(folders))
    return list(dict.fromkeys(t for t in texts if t))


def chosen_regions(instrument: dict, notes: int = NOTES) -> list:
    """A few notes spread over the instrument's range; for each, its middle and its loudest velocity layer
    (first round robin)."""
    attack = [r for r in instrument["regions"] if r["trigger"] == "attack" and r.get("round_robin", 1) == 1]
    keys = sorted({r["key"] for r in attack})
    picked = sorted({keys[round(i * (len(keys) - 1) / max(1, notes - 1))] for i in range(min(notes, len(keys)))})
    out = []
    for key in picked:
        layers = sorted((r for r in attack if r["key"] == key), key=lambda r: r["hivel"])
        out += [layers[len(layers) // 2]] + ([layers[-1]] if len(layers) > 1 else [])
    return list({r["file"]: r for r in out}.values())


def load(path: Path) -> np.ndarray:
    import soundfile as sf
    import soxr

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True, frames=int(SECONDS * 96000))
    mono = audio.mean(axis=1)
    if rate != SR:
        mono = soxr.resample(mono, rate, SR, quality="HQ")
    return mono[:int(SECONDS * SR)].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="libraries, catalog.json and packs/ (build_packs.py)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    from src.v2.clap_audio import ClapEmbedder

    original = {i["id"]: i for i in json.loads((args.root / "catalog.json").read_text(encoding="utf-8"))["instruments"]}
    full = json.loads((args.root / "packs" / "catalog-full.json").read_text(encoding="utf-8"))["instruments"]
    waves, owners = [], []
    for instrument in full:
        for region in chosen_regions(instrument):
            waves.append(load(args.root / "packs" / "full" / region["file"]))
            owners.append(instrument["id"])
    emb = ClapEmbedder(half=True).embed_audio(waves, batch_size=32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    texts, audio, ids = [], [], []
    by_id = {i["id"]: i for i in full}
    for e, owner in zip(emb, owners):
        for text in names(by_id[owner], original[owner]):
            texts.append(text), audio.append(e), ids.append(owner)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, texts=np.array(texts), emb=np.stack(audio).astype(np.float32), ids=np.array(ids))
    print(f"{len(full)} instruments, {len(waves)} recordings heard, {len(texts)} pairs, "
          f"{len(set(texts))} names -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
