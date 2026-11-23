#!/usr/bin/env python3
"""Text for the v4 text encoder and prior: the v3 corpus plus the uploaders' own descriptions (#35).

The v3 encoder learnt from titles, tags, class labels and preset names: short phrases. Users type sentences
("verre qui se casse net"). FSD50K keeps each uploader's description of the sound; once cleaned
(src/v4/descriptions.py), they are human sentences about real recordings. Only recordings under CC0 or CC BY
are used, and only the development split: the eval split stays held out for scripts/v4/text_gap.py.

    python scripts/v4/build_corpus_v4.py                 # -> data/v4/descriptions.jsonl, corpus_todo.jsonl
    python scripts/v2/translate_corpus.py --inp data/v4/corpus_todo.jsonl --out data/v4/corpus_todo_fr.jsonl
    python scripts/v4/build_corpus_v4.py --merge         # -> data/v4/corpus_en_fr.jsonl (+ French descriptions)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.v4.descriptions import clean_description, shared_sentences, usable, without  # noqa: E402

OUT_DIR = BASE_DIR / "data" / "v4"
V3_CORPUS = BASE_DIR / "data" / "v3" / "corpus_en_fr.jsonl"
FREE = ("/publicdomain/zero/", "/licenses/by/")  # CC0, CC BY (any version); not NC, not Sampling+


def free(licence: str) -> bool:
    return any(k in licence for k in FREE)


def benchmark_texts() -> set:
    """Malo's benchmark prompts and their judge wording: never trained on."""
    texts = set()
    for name in ("v3_prompts_p1.json", "v3_prompts_p2.json"):
        for p in json.loads((BASE_DIR / "benchmarks" / name).read_text(encoding="utf-8")):
            texts |= {p.lower()} if isinstance(p, str) else {p["prompt"].lower(), p["judge"].lower()}
    return texts


def descriptions(fsd50k: Path):
    """One row per CC0 / CC BY development recording whose description says something about the sound."""
    import csv

    info = json.loads((fsd50k / "FSD50K.metadata" / "dev_clips_info_FSD50K.json").read_text(encoding="utf-8"))
    with open(fsd50k / "FSD50K.ground_truth" / "dev.csv", newline="", encoding="utf-8") as f:
        labels = {r["fname"]: r["labels"] for r in csv.DictReader(f)}
    cleaned = {fname: clean_description(v.get("description")) for fname, v in info.items() if free(v["license"])}
    shared = shared_sentences(cleaned.values())
    held_out = benchmark_texts()
    rows = []
    for fname, text in sorted(cleaned.items(), key=lambda kv: int(kv[0])):
        text = without(text, shared)
        if usable(text) and text.lower() not in held_out:
            v = info[fname]
            rows.append({"fname": fname, "en": text, "labels": labels.get(fname, ""), "license": v["license"],
                         "uploader": v["uploader"]})
    return rows, len(cleaned), len(shared)


def merge(out: Path) -> int:
    translated = {}
    for line in open(out / "corpus_todo_fr.jsonl", encoding="utf-8"):
        row = json.loads(line)
        if row.get("fr"):
            translated[row["en"]] = row["fr"]
    rows = [json.loads(line) for line in open(out / "descriptions.jsonl", encoding="utf-8")]
    for row in rows:
        row["fr"] = translated.get(row["en"], "")
    with open(out / "descriptions.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    added = 0
    with open(out / "corpus_en_fr.jsonl", "w", encoding="utf-8") as f:
        for line in open(V3_CORPUS, encoding="utf-8"):
            f.write(line if line.endswith("\n") else line + "\n")
        for en, fr in sorted(translated.items()):
            f.write(json.dumps({"en": en, "source": "fsd50k_description", "fr": fr}, ensure_ascii=False) + "\n")
            added += 1
    print(json.dumps({"descriptions": len(rows), "translated": sum(bool(r["fr"]) for r in rows),
                      "corpus_sentences_added": added}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsd50k", type=Path, default=Path.home() / "Datasets" / "FSD50K")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--merge", action="store_true", help="after translation: write corpus_en_fr.jsonl")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        return merge(args.out)
    rows, free_clips, shared = descriptions(args.fsd50k)
    with open(args.out / "descriptions.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    unique = sorted({r["en"] for r in rows})
    with open(args.out / "corpus_todo.jsonl", "w", encoding="utf-8") as f:
        for text in unique:
            f.write(json.dumps({"en": text, "source": "fsd50k_description"}, ensure_ascii=False) + "\n")
    print(json.dumps({"cc0_or_by_recordings": free_clips, "shared_sentences_removed": shared,
                      "recordings_with_description": len(rows), "unique_descriptions": len(unique)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
