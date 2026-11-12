#!/usr/bin/env python3
"""Translate the English corpus to French with Helsinki-NLP/opus-mt-en-fr (resumable).

    python scripts/v2/translate_corpus.py --inp data/v2/corpus_en.jsonl --out data/v2/corpus_en_fr.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

BASE_DIR = Path(__file__).resolve().parents[2]
TRANSLATOR_ID = "Helsinki-NLP/opus-mt-en-fr"
TRANSLATOR_REVISION = "dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18"

# Sound-design jargon the general-purpose translator gets wrong. French producers keep most of
# these English words ("un lead saturé", "une nappe", "un kick"), so the fix keeps or adapts them.
# (English word present, wrong French rendering, replacement)
GLOSSARY = [
    (r"\blead\b(?!\s+pipe)", r"\bplombs?\b", "lead"),
    (r"\bpads?\b", r"\b(tampons?|coussinets?|blocs?|patins?|bloc-notes)\b", "nappe"),
    (r"\bsnares?\b", r"\b(pi[eè]ges?|collets?)\b", "caisse claire"),
    (r"\bkicks?\b", r"\bcoups? de pied\b", "kick"),
    (r"\bplucks?\b|\bplucky\b", r"\b(cueillir|cueillette|plumes?|arrach\w*)\b", "pluck"),
    (r"\bstabs?\b", r"\bcoups? de (couteau|poignard)\b", "stab"),
    (r"\brisers?\b", r"\b(colonnes? montantes?|l[eè]ve-t[oô]t)\b", "riser"),
    (r"\bbrass\b", r"\blaiton\b", "cuivres"),
    (r"\bkeys\b", r"\bcl[eé]s\b", "claviers"),
    (r"\bdrop\b", r"\bgouttes?\b", "drop"),
    (r"\bhi-?hats?\b", r"\bchapeaux? (haut|hauts)\b", "charleston"),
]


def fix_jargon(en: str, fr: str) -> str:
    for en_pattern, fr_bad, fr_good in GLOSSARY:
        if re.search(en_pattern, en, flags=re.I):
            fr = re.sub(fr_bad, fr_good, fr, flags=re.I)
    return fr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inp", type=Path, default=BASE_DIR / "data" / "v2" / "corpus_en.jsonl")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "data" / "v2" / "corpus_en_fr.jsonl")
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--device", default=os.environ.get("HARMONIA_TRANSLATE_DEVICE", "cpu"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-tags", type=int, default=10000, help="cap on fsd50k_tags sentences")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    rows = [json.loads(line) for line in args.inp.read_text(encoding="utf-8").splitlines() if line.strip()]
    tags = [r for r in rows if r["source"] == "fsd50k_tags"]
    if len(tags) > args.max_tags:
        keep = {r["en"] for r in random.Random(0).sample(tags, args.max_tags)}  # nosec B311 - seeded RNG for reproducible data, not security-sensitive
        rows = [r for r in rows if r["source"] != "fsd50k_tags" or r["en"] in keep]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["en"] for line in args.out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [r for r in rows if r["en"] not in done]
    print(f"{len(rows)} sentences, {len(done)} already translated, {len(todo)} to go", flush=True)

    tokenizer = MarianTokenizer.from_pretrained(TRANSLATOR_ID, revision=TRANSLATOR_REVISION)  # nosec B615
    model = MarianMTModel.from_pretrained(TRANSLATOR_ID, revision=TRANSLATOR_REVISION).to(args.device).eval()  # nosec B615
    start = time.time()
    with open(args.out, "a", encoding="utf-8") as f, torch.no_grad():
        for i in range(0, len(todo), args.batch):
            chunk = todo[i : i + args.batch]
            enc = tokenizer([r["en"] for r in chunk], return_tensors="pt", padding=True, truncation=True, max_length=64)
            enc = {k: v.to(args.device) for k, v in enc.items()}
            max_new = min(64, int(enc["input_ids"].shape[1]) * 2 + 8)
            out = model.generate(**enc, num_beams=2, max_new_tokens=max_new, no_repeat_ngram_size=3)
            for row, fr in zip(chunk, tokenizer.batch_decode(out, skip_special_tokens=True)):
                fr = fix_jargon(row["en"], fr)
                if len(fr.split()) > 3 * len(row["en"].split()) + 4:
                    continue  # degenerate translation
                f.write(json.dumps({**row, "fr": fr}, ensure_ascii=False) + "\n")
            f.flush()
            n = i + len(chunk)
            if (i // args.batch) % 20 == 0:
                rate = n / max(1e-6, time.time() - start)
                print(f"{n}/{len(todo)} | {rate:.1f} sent/s | eta {(len(todo) - n) / max(rate, 1e-6) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
