#!/usr/bin/env python3
"""How close is the sound a text asks for to the real sound? (#35)

The run time turns a prompt into a target in CLAP's audio space (text encoder -> prior), then generators
make presets for that target. This measures the first half alone, without playing anything, on texts the
model never saw: FSD50K's eval split.

- descriptions: the uploaders' cleaned descriptions (src/v4/descriptions.py) of 2,000 eval recordings;
  target vs the recording itself: cosine, and whether the recording is in the top 10 of the 2,000;
- descriptions_fr: the same in French (machine translation, cached in data/v4/eval_descriptions_fr.jsonl);
- classes: the eval class names ("Glass", "Slam"...), human voice classes left out; target vs the centroid
  of the class's eval recordings: cosine, and whether the right class is in the top 5.

    python scripts/v4/text_gap.py --model-dir models/harmonia_v3
    python scripts/v4/text_gap.py --student data/v4/text_student --prior data/v4/generator/prior.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.v4.descriptions import clean_description, usable  # noqa: E402

FSD50K_EVAL = next((p for p in (BASE_DIR / "data" / "v2" / "fsd50k_eval.npz",
                                BASE_DIR.parent / ".worktrees" / "ai-v2" / "HARMONIA" / "data" / "v2" / "fsd50k_eval.npz")
                    if p.exists()), BASE_DIR / "data" / "v2" / "fsd50k_eval.npz")
FRENCH_CACHE = BASE_DIR / "data" / "v4" / "eval_descriptions_fr.jsonl"
SAMPLE, SEED, FRENCH = 2000, 0, 500


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def eval_set(fsd50k: Path, with_fnames: bool = False):
    """(texts, recording embeddings, class names, class centroids[, recording names]) of the eval split."""
    from scripts.v3.match_recordings import VOICE

    z = np.load(FSD50K_EVAL)
    emb = normalize(z["emb"].astype(np.float32))
    fnames = [str(f) for f in z["fname"]]
    info = json.loads((fsd50k / "FSD50K.metadata" / "eval_clips_info_FSD50K.json").read_text(encoding="utf-8"))
    texts = [clean_description(info[f]["description"]) for f in fnames]
    keep = np.array([i for i, t in enumerate(texts) if usable(t)])
    pick = np.random.default_rng(SEED).choice(keep, SAMPLE, replace=False)
    with open(fsd50k / "FSD50K.ground_truth" / "eval.csv", newline="", encoding="utf-8") as f:
        leaf = {r["fname"]: r["labels"].split(",")[0] for r in csv.DictReader(f)}
    names = sorted(set(leaf.values()) - VOICE)
    own = np.array([leaf.get(f, "") for f in fnames])
    centroids = normalize(np.stack([emb[own == n].mean(0) for n in names]))
    picked = ([texts[i] for i in pick], emb[pick], names, centroids)
    return (*picked, [fnames[i] for i in pick]) if with_fnames else picked


def french(texts):
    """French versions of the first FRENCH texts, translated once and cached."""
    cached = {}
    if FRENCH_CACHE.exists():
        cached = {r["en"]: r["fr"] for r in map(json.loads, FRENCH_CACHE.read_text(encoding="utf-8").splitlines())}
    todo = [t for t in texts[:FRENCH] if t not in cached]
    if todo:
        from transformers import MarianMTModel, MarianTokenizer

        from scripts.v2.translate_corpus import TRANSLATOR_ID, TRANSLATOR_REVISION, fix_jargon

        tok = MarianTokenizer.from_pretrained(TRANSLATOR_ID, revision=TRANSLATOR_REVISION)  # nosec B615
        model = MarianMTModel.from_pretrained(TRANSLATOR_ID, revision=TRANSLATOR_REVISION).eval()  # nosec B615
        FRENCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(FRENCH_CACHE, "a", encoding="utf-8") as f, torch.no_grad():
            for i in range(0, len(todo), 32):
                chunk = todo[i:i + 32]
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=64)
                out = model.generate(**enc, num_beams=2, max_new_tokens=64, no_repeat_ngram_size=3)
                for en, fr in zip(chunk, tok.batch_decode(out, skip_special_tokens=True)):
                    cached[en] = fix_jargon(en, fr)
                    f.write(json.dumps({"en": en, "fr": cached[en]}, ensure_ascii=False) + "\n")
    return [cached[t] for t in texts[:FRENCH]]


def report(targets: np.ndarray, real: np.ndarray, answer: np.ndarray, top: int) -> dict:
    """Mean cosine to the right sound, how often it is in the top `top` of all of them, its median rank."""
    sims = targets @ real.T
    right = sims[np.arange(len(targets)), answer]
    rank = (sims > right[:, None]).sum(1)
    return {"cos": round(float(right.mean()), 4), f"top{top}": round(float((rank < top).mean()), 4),
            "median_rank": int(np.median(rank))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=BASE_DIR / "models" / "harmonia_v3")
    parser.add_argument("--student", type=Path, default=None, help="text encoder folder (default: the model's)")
    parser.add_argument("--prior", type=Path, default=None, help="prior weights (default: the model's)")
    parser.add_argument("--fsd50k", type=Path, default=Path.home() / "Datasets" / "FSD50K")
    parser.add_argument("--teacher", action="store_true", help="CLAP's own text encoder instead of the student")
    parser.add_argument("--out", type=Path, default=None, help="write the scores to this JSON file")
    args = parser.parse_args()

    from src.v2.text_student import encode_texts, load_trained_student
    from src.v3.prior import Prior

    device = torch.device("cpu")
    prior = Prior()
    prior.load_state_dict(torch.load(args.prior or args.model_dir / "generator" / "prior.pt", map_location="cpu",
                                     weights_only=True))
    prior.eval()
    if args.teacher:
        from src.v2.clap_audio import ClapEmbedder

        teacher = ClapEmbedder(half=False)

        def encode(texts):
            return teacher.embed_text(list(texts))
    else:
        student, tokenizer = load_trained_student(args.student or args.model_dir / "text_student", device)

        def encode(texts):
            return np.concatenate([encode_texts(student, tokenizer, list(texts[i:i + 256]), device)
                                   for i in range(0, len(texts), 256)])

    def target(texts):
        with torch.no_grad():
            return normalize(prior(torch.from_numpy(np.asarray(encode(texts), dtype=np.float32))).numpy())

    texts, real, names, centroids = eval_set(args.fsd50k)
    result = {
        "descriptions": report(target(texts), real, np.arange(len(texts)), 10),
        "descriptions_fr": report(target(french(texts)), real[:FRENCH], np.arange(FRENCH), 10),
        "classes": report(target([n.replace("_", " ").lower() for n in names]), centroids, np.arange(len(names)), 5),
    }
    result["descriptions_fr"]["note"] = f"first {FRENCH} descriptions, ranked among {FRENCH}"
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
