#!/usr/bin/env python3
"""Train the v4 text -> sound prior (#35).

Same model and loss as v3 (src/v3/prior.py), more human pairs:
- v3's pairs: FSD50K development recordings with their title and class label, bank presets with their own
  names and categories;
- v4 adds the uploaders' cleaned descriptions (scripts/v4/build_corpus_v4.py): sentences about the sound,
  paired with the recording they describe.

`--free-only` keeps FSD50K pairs of CC0 and CC BY recordings only (v3 also used CC BY-NC and Sampling+).
`--min-agreement` drops description pairs whose text and sound CLAP finds unrelated (an uploader writing
about the weather while recording traffic); 0 keeps them all.

`--instruments` (#47) adds the names the sample libraries give their instruments, paired with the sound of
their own recordings (scripts/samples/instrument_texts.py): the prior then knows what "Solo Violin" or "Cello
Section" sound like. They are few next to the other pairs, so each is seen `--instrument-repeat` times.

    python scripts/v4/train_prior_v4.py --out data/v4/prior/desc.pt
    python scripts/v4/text_gap.py --prior data/v4/prior/desc.pt
    python scripts/v4/train_prior_v4.py --out data/samples/prior/names10.pt --free-only --min-agreement 0.10 \
        --instruments data/samples/instrument_pairs.npz --instrument-repeat 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.v2.build_text_corpus import clean_title, label_phrase  # noqa: E402
from scripts.v3.train_generator_v3 import FSD50K_DEV, load_bank, normalize, train_prior  # noqa: E402
from scripts.v4.build_corpus_v4 import free  # noqa: E402
from src.presets.labels import preset_text  # noqa: E402

DESCRIPTIONS = BASE_DIR / "data" / "v4" / "descriptions.jsonl"


def pairs(free_only: bool, descriptions: bool, fsd50k: Path, instruments: Path | None = None, repeat: int = 1):
    """(texts, audio embeddings, kinds) of every human pair."""
    info = json.loads((fsd50k / "FSD50K.metadata" / "dev_clips_info_FSD50K.json").read_text(encoding="utf-8"))
    fsd = np.load(FSD50K_DEV)
    fsd_emb = normalize(fsd["emb"].astype(np.float32))
    row_of = {str(f): i for i, f in enumerate(fsd["fname"])}
    texts, audio, kinds = [], [], []
    for fname, title, labels, e in zip(fsd["fname"], fsd["title"], fsd["labels"], fsd_emb):
        if free_only and not free(info[str(fname)]["license"]):
            continue
        title = clean_title(str(title))
        if len(title.split()) >= 2:
            texts.append(title), audio.append(e), kinds.append("title")
        texts.append(label_phrase(str(labels).split(",")[0])), audio.append(e), kinds.append("label")
    if descriptions:
        for row in map(json.loads, DESCRIPTIONS.read_text(encoding="utf-8").splitlines()):
            if row["fname"] in row_of:  # always CC0 / CC BY (build_corpus_v4.py)
                texts.append(row["en"]), audio.append(fsd_emb[row_of[row["fname"]]]), kinds.append("description")
    _, entries, emb = load_bank()
    for entry, e in zip(entries, emb.mean(axis=1)):
        text = preset_text(entry)
        if text:
            texts.append(text), audio.append(e), kinds.append("preset")
    if instruments is not None:
        z = np.load(instruments)
        for _ in range(repeat):
            for text, e in zip(z["texts"], z["emb"]):
                texts.append(str(text)), audio.append(e), kinds.append("instrument")
    return texts, normalize(np.stack(audio)).astype(np.float32), np.array(kinds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="prior weights (.pt); a .json report goes next to it")
    parser.add_argument("--fsd50k", type=Path, default=Path.home() / "Datasets" / "FSD50K")
    parser.add_argument("--free-only", action="store_true", help="FSD50K pairs of CC0 / CC BY recordings only")
    parser.add_argument("--no-descriptions", action="store_true", help="v3's pairs only (to compare)")
    parser.add_argument("--min-agreement", type=float, default=0.0,
                        help="drop description pairs whose CLAP text-sound cosine is below this")
    parser.add_argument("--instruments", type=Path, default=None,
                        help="names of the recorded instruments and their sound (scripts/samples/instrument_texts.py)")
    parser.add_argument("--instrument-repeat", type=int, default=10, help="times each instrument pair is seen")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    from src.v2.clap_audio import ClapEmbedder, best_device

    texts, audio, kinds = pairs(args.free_only, not args.no_descriptions, args.fsd50k, args.instruments,
                                args.instrument_repeat)
    unique = sorted(set(texts))
    embedder = ClapEmbedder(half=True)
    table = dict(zip(unique, normalize(embedder.embed_text(unique).astype(np.float32))))
    del embedder
    text_emb = np.stack([table[t] for t in texts]).astype(np.float32)
    agreement = (text_emb * audio).sum(1)
    keep = (kinds != "description") | (agreement >= args.min_agreement)
    counts = {k: int(((kinds == k) & keep).sum()) for k in ("title", "label", "description", "preset", "instrument")}
    print("pairs", json.dumps(counts), flush=True)
    prior, report = train_prior(text_emb[keep], audio[keep], args.epochs, 512, best_device(), args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prior.state_dict(), args.out)
    meta = {"pairs": counts, "dropped_descriptions": int(((kinds == "description") & ~keep).sum()),
            "free_only": args.free_only, "min_agreement": args.min_agreement, "epochs": args.epochs,
            "instruments": str(args.instruments) if args.instruments else None,
            "instrument_repeat": args.instrument_repeat if args.instruments else 0,
            "seed": args.seed, "report": report}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
