#!/usr/bin/env python3
"""Does the analog generator pick the right recorded instrument for a text? (#47)

Texts: the titles uploaders gave to recordings of instruments in FSD50K's eval split (never learnt; CC0 and CC BY
recordings only), cleaned like the training titles. Each title goes through the run time's text side (text
encoder, prior) and the analog generator draws presets for it; a draw counts as right when it plays a recorded
instrument of the family of the recording's class (a violin recording's title should draw strings).

No sound is played: this measures the choice of instrument, quickly, for several priors.

    python scripts/samples/instrument_accuracy.py --model-dir data/samples/models/s1 --name s1
    python scripts/samples/instrument_accuracy.py --model-dir data/samples/models/s1 --prior data/samples/prior/names.pt --name names
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.v2.build_text_corpus import clean_title  # noqa: E402
from scripts.v3.match_recordings import LICENSES  # noqa: E402

# FSD50K class (most specific label) -> family of the sample catalogue (samples/instruments.json)
CLASS_FAMILY = {"Bowed_string_instrument": "strings", "Wind_instrument_and_woodwind_instrument": "woodwinds",
                "Harmonica": "woodwinds", "Brass_instrument": "brass", "Trumpet": "brass", "Piano": "keys",
                "Organ": "organ", "Harp": "plucked", "Plucked_string_instrument": "plucked",
                "Marimba_and_xylophone": "mallets", "Glockenspiel": "mallets", "Mallet_percussion": "mallets",
                "Bell": "mallets", "Church_bell": "mallets"}
OUT = BASE_DIR / "data" / "samples" / "accuracy"


def eval_titles(fsd50k: Path, per_class: int, seed: int = 0):
    """[(title, class)] of instrument recordings of the eval split, up to per_class per class."""
    info = json.loads((fsd50k / "FSD50K.metadata" / "eval_clips_info_FSD50K.json").read_text(encoding="utf-8"))
    by_class = defaultdict(list)
    with open(fsd50k / "FSD50K.ground_truth" / "eval.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            leaf = row["labels"].split(",")[0]
            meta = info.get(row["fname"], {})
            if leaf not in CLASS_FAMILY or not any(k in meta.get("license", "") for k in LICENSES):
                continue
            title = clean_title(str(meta.get("title", "")))
            if len(title.split()) >= 2:
                by_class[leaf].append(title)
    rng = random.Random(seed)  # nosec B311 - picks evaluation texts, not security-sensitive
    out = []
    for leaf in sorted(by_class):
        titles = sorted(set(by_class[leaf]))
        rng.shuffle(titles)
        out += [(t, leaf) for t in titles[:per_class]]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="an exported model with the sample source")
    parser.add_argument("--prior", type=Path, default=None, help="other prior weights to try with it")
    parser.add_argument("--name", required=True)
    parser.add_argument("--fsd50k", type=Path, default=Path.home() / "Datasets" / "FSD50K")
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--draws", type=int, default=8)
    args = parser.parse_args()

    import torch

    from src.samples.player import instrument_ids
    from src.v3 import codecs
    from src.v3.inference import HarmoniaV3, Settings

    family = {i["id"]: i["family"] for i in
              json.loads((BASE_DIR / "samples" / "instruments.json").read_text(encoding="utf-8"))["instruments"]}
    ids = instrument_ids()
    model = HarmoniaV3(args.model_dir, settings=Settings(), listener=object())
    if args.prior:
        model.prior.load_state_dict(torch.load(args.prior, map_location="cpu", weights_only=True))
    if model.dims["analog"] <= codecs.ANALOG_DIM_V3:
        raise SystemExit("this model's analog generator has no sample source")
    rows = []
    for k, (title, leaf) in enumerate(eval_titles(args.fsd50k, args.per_class)):
        drawn = model._draw("analog", model.target(title), args.draws, seed=k)
        choices = [int(codecs.analog_decode(v, model.timbres)[codecs.INSTRUMENT]) for v in drawn]
        families = [family[ids[c]] if c else None for c in choices]
        rows.append({"title": title, "class": leaf, "families": families,
                     "right": float(np.mean([f == CLASS_FAMILY[leaf] for f in families])),
                     "recorded": float(np.mean([f is not None for f in families]))})
    classes = sorted({r["class"] for r in rows})
    per_class = {c: {"titles": sum(r["class"] == c for r in rows),
                     "right": round(float(np.mean([r["right"] for r in rows if r["class"] == c])), 3),
                     "recorded": round(float(np.mean([r["recorded"] for r in rows if r["class"] == c])), 3)}
                 for c in classes}
    summary = {"name": args.name, "titles": len(rows),
               "right": round(float(np.mean([v["right"] for v in per_class.values()])), 3),
               "recorded": round(float(np.mean([v["recorded"] for v in per_class.values()])), 3),
               "per_class": per_class}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.name}.json").write_text(json.dumps({**summary, "rows": rows}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    for c, v in per_class.items():
        print(f"  {c[:32]:32s} {v['titles']:3d} titles  right family {v['right']:.0%}  recorded {v['recorded']:.0%}")
    print(f"{args.name}: {len(rows)} titles, right family {summary['right']:.0%} (mean over classes), "
          f"recorded instrument {summary['recorded']:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
