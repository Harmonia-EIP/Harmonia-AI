#!/usr/bin/env python3
"""Timbre coordinates of every sampled instrument (src/samples/descriptors.py, #46).

Each instrument is played alone through the analog engine (sample level 1, everything else at its default:
filter open, no effect) at the generators' notes, from the "listen" pack the server plays; CLAP listens and
the principal axes of the 125 sounds give each instrument its coordinates.

    python scripts/samples/instrument_descriptors.py            # -> samples/descriptors.json
    python scripts/samples/instrument_descriptors.py --check    # the table describes every charter instrument

Also writes data/samples/instruments_clap.npz (every instrument's full CLAP vector, not in git): the imitation
search (scripts/v3/match_recordings.py --samples) uses it to pick the instruments worth trying for a recording.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src import charter_v3  # noqa: E402
from src.samples import descriptors  # noqa: E402

NOTES = (45, 69)  # the notes the generators hear their presets at (src/v3/inference.py)
HOLD_SECONDS, TOTAL_SECONDS, SR, VELOCITY = 1.5, 4.0, 48000, 100
CLAP_OUT = BASE_DIR / "data" / "samples" / "instruments_clap.npz"


def solo_presets(choices) -> np.ndarray:
    """One preset per instrument: the instrument alone, every other setting at its default."""
    from src.synth import v3_params as P

    presets = np.repeat(P.DEFAULTS[None, :], len(choices), axis=0)
    presets[:, P.INDEX["sample_instrument"]] = choices
    presets[:, P.INDEX["sample_level"]] = 1.0
    return presets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=BASE_DIR / "models" / "samples",
                        help="an unpacked sample pack (the listen pack, like the server)")
    parser.add_argument("--dims", type=int, default=16,
                        help="16: 85 % of what CLAP hears, the nearest instrument among CLAP's 3 nearest for 96 %")
    parser.add_argument("--out", type=Path, default=descriptors.PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    ids = charter_v3.load().raw["sample_instruments"]["ids"]
    if args.check:
        missing = [i for i in ids if i not in json.loads(args.out.read_text(encoding="utf-8"))["instruments"]]
        print("descriptors up to date" if not missing else f"not described: {missing}")
        return 1 if missing else 0

    from src.samples.player import SampleBank
    from src.synth.engine_v3 import render_batch
    from src.v2.clap_audio import TEACHER_ID, TEACHER_REVISION, ClapEmbedder

    bank = SampleBank(args.samples)
    choices = [c for c in range(1, len(ids) + 1) if bank.has(c)]
    audio = np.nan_to_num(render_batch(solo_presets(choices), np.array(NOTES), VELOCITY / 127, HOLD_SECONDS,
                                       TOTAL_SECONDS, SR, 7, bank))
    emb = ClapEmbedder(half=True).embed_audio(list(audio.reshape(-1, audio.shape[-1]).astype(np.float32)),
                                              batch_size=32)
    emb = emb.reshape(len(choices), len(NOTES), -1).mean(axis=1)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    described = [ids[c - 1] for c in choices]
    CLAP_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CLAP_OUT, ids=np.array(described), emb=emb.astype(np.float32))

    fitted = descriptors.fit(emb, described, args.dims)
    kept = descriptors.neighbours_kept(emb, np.array(list(fitted["instruments"].values())))
    raw = {"version": 1, "description": descriptors.__doc__.strip().splitlines()[0],
           "listener": {"id": TEACHER_ID, "revision": TEACHER_REVISION}, "notes": list(NOTES),
           "pack": "listen", **fitted, "nearest_neighbour_kept": round(kept, 3)}
    text = json.dumps(raw, indent=1, ensure_ascii=False)
    text = re.sub(r"\[\s+([^\[\]]+?)\s+\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", text)  # a row per line
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"{len(described)} instruments, {args.dims} coordinates: {fitted['explained_variance']:.0%} of what CLAP "
          f"hears, nearest instrument kept for {kept:.0%} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
