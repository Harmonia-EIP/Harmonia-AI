#!/usr/bin/env python3
"""Human-written English text for the v3 text encoder (no generated sentences).

- FSD50K titles, tags and class labels (written by the people who uploaded or annotated the sounds),
  taken with their French translation from the v2 corpus; v2's template and concept sentences are dropped.
- Real presets' own words: names (abbreviations expanded), categories and comments (src/presets/labels.py).

    python scripts/v3/build_text_corpus_v3.py   # -> data/v3/corpus_known.jsonl (en + fr), corpus_todo.jsonl (en)
    python scripts/v2/translate_corpus.py --inp data/v3/corpus_todo.jsonl --out data/v3/corpus_todo_fr.jsonl
    python scripts/v3/build_text_corpus_v3.py --merge   # -> data/v3/corpus_en_fr.jsonl

Preset names are mostly proper nouns ("laurie", "beatmehrdr") that the translator mangles ("laurier"):
a preset text keeps its machine translation only when 75 % of its words are descriptive vocabulary
(frequent FSD50K words, instrument words); otherwise its French text is the name itself.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.presets.filters import playable  # noqa: E402
from src.presets.labels import INSTRUMENT_WORDS, expand, preset_text  # noqa: E402

HUMAN_V2_SOURCES = {"fsd50k_title", "fsd50k_tags", "fsd50k_label"}
OUT_DIR = BASE_DIR / "data" / "v3"


def merge(out: Path) -> int:
    known = [json.loads(line) for line in open(out / "corpus_known.jsonl", encoding="utf-8")]
    todo = [json.loads(line) for line in open(out / "corpus_todo_fr.jsonl", encoding="utf-8")]
    counts = collections.Counter(w for r in known for w in re.findall(r"[a-z]+", r["en"].lower()))
    vocabulary = {w for w, c in counts.items() if c >= 5} | INSTRUMENT_WORDS
    names = 0
    for row in todo:
        words = re.findall(r"[a-z]+", row["en"].lower())
        if not words or sum(w in vocabulary for w in words) / len(words) < 0.75:
            row["fr"] = row["en"]
            names += 1
    with open(out / "corpus_en_fr.jsonl", "w", encoding="utf-8") as f:
        for row in known + todo:
            if row.get("fr"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(known) + len(todo), "preset_names_kept": names}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-corpus", type=Path,
                        default=BASE_DIR.parents[1] / "ai-v2" / "HARMONIA" / "data" / "v2" / "corpus_en_fr.jsonl")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--merge", action="store_true", help="merge the translated preset texts")
    args = parser.parse_args()
    if args.merge:
        return merge(args.out)

    known, todo, seen = [], [], set()
    for line in open(args.v2_corpus, encoding="utf-8"):
        row = json.loads(line)
        if row["source"] in HUMAN_V2_SOURCES and row["en"].lower() not in seen:
            seen.add(row["en"].lower())
            known.append(row)
    for source in ("dx7", "obxf", "surge"):
        for line in open(BASE_DIR / "data" / "v3" / "presets" / f"{source}.jsonl", encoding="utf-8"):
            rec = json.loads(line)
            if source != "dx7" and not playable(rec):
                continue
            texts = [preset_text(rec), expand(rec["name"])]
            if rec.get("category"):
                texts.append(expand(str(rec["category"]).replace("\\", " ").replace("/", " ")))
            for text in texts:
                text = text.strip()
                if 2 <= len(text) <= 160 and text.lower() not in seen:
                    seen.add(text.lower())
                    todo.append({"en": text, "source": f"preset_{source}"})
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("corpus_known.jsonl", known), ("corpus_todo.jsonl", todo)):
        with open(args.out / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for row in known + todo:
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    print(json.dumps({"translated": len(known), "to_translate": len(todo), **counts}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
