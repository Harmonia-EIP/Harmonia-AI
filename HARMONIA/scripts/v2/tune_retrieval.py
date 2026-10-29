#!/usr/bin/env python3
"""Tune the retrieval settings on FSD50K *eval* (never used for training, nor by the benchmark).

For each class of the eval split, the query is its label phrase (English and French); a setting
is good when the preset it retrieves sounds like the real recordings of that class (cosine
between the preset's CLAP audio embedding and the class centroid of real clips).

    python scripts/v2/tune_retrieval.py --data data/v2 [--with-predictor]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.v2.build_text_corpus import label_phrase  # noqa: E402
from scripts.v2.train_predictor import LOSS_WEIGHTS, load_bank  # noqa: E402
from src.charter import PARAM_NAMES  # noqa: E402
from src.v2.clap_audio import best_device  # noqa: E402
from src.v2.retrieval import AUDIBLE_DB_FLOOR, top_indices  # noqa: E402
from src.v2.text_student import encode_texts, load_trained_student  # noqa: E402


def class_centroids(fsd_eval):
    groups = defaultdict(list)
    for labels, emb in zip(fsd_eval["labels"], fsd_eval["emb"].astype(np.float32)):
        groups[str(labels).split(",")[0]].append(emb)
    out = {}
    for cls, embs in groups.items():
        if len(embs) >= 10:
            c = np.mean(embs, axis=0)
            out[cls] = c / np.linalg.norm(c)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "v2")
    parser.add_argument("--with-predictor", action="store_true", help="also tune the hybrid weight")
    args = parser.parse_args()

    bank_params, bank_emb = load_bank(args.data / "bank" / "bank.npz")
    extra = np.load(args.data / "bank" / "bank_extra.npz")
    audible = extra["audible_db"] >= AUDIBLE_DB_FLOOR
    centroids = class_centroids(np.load(args.data / "fsd50k_eval.npz"))
    classes = sorted(centroids)

    fr_by_en = {}
    for line in (args.data / "corpus_en_fr.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("fr"):
            fr_by_en[row["en"]] = row["fr"]
    queries = {"en": [label_phrase(c) for c in classes]}
    queries["fr"] = [fr_by_en.get(q, q) for q in queries["en"]]

    device = best_device()
    student, tokenizer = load_trained_student(args.data / "text_student", device)
    target = np.stack([centroids[c] for c in classes])
    results = []
    for filter_quiet in (False, True):
        emb = bank_emb if not filter_quiet else bank_emb[audible]
        params = bank_params if not filter_quiet else bank_params[audible]
        hub = extra["hub"] if not filter_quiet else extra["hub"][audible]
        for lang, texts in queries.items():
            q = encode_texts(student, tokenizer, texts, device)
            for csls in (0.0, 0.25, 0.5, 0.75, 1.0):
                picks = top_indices(q, emb, hub, csls, k=1)[:, 0]
                score = float((emb[picks] * target).sum(axis=1).mean())
                results.append({"filter_quiet": filter_quiet, "lang": lang, "csls_weight": csls, "score": round(score, 4)})
                if args.with_predictor:
                    import torch

                    from src.v2.predictor import ParamPredictor

                    model = ParamPredictor()
                    model.load_state_dict(torch.load(args.data / "predictor" / "predictor.pt", map_location="cpu", weights_only=True))
                    model.eval()
                    with torch.no_grad():
                        neural = model(torch.from_numpy(q).float())[0].numpy()
                    weights = np.array([LOSS_WEIGHTS[n] for n in PARAM_NAMES], dtype=np.float32)
                    top = top_indices(q, emb, hub, csls, k=32)
                    sims = np.take_along_axis(q @ emb.T, top, axis=1) - csls * hub[top]
                    dist = np.sqrt((((params[top] - neural[:, None, :]) ** 2) * weights).mean(axis=2))
                    for hybrid in (0.05, 0.1, 0.15, 0.25, 0.5):
                        pick = top[np.arange(len(top)), np.argmax(sims - hybrid * dist, axis=1)]
                        score = float((emb[pick] * target).sum(axis=1).mean())
                        results.append({"filter_quiet": filter_quiet, "lang": lang, "csls_weight": csls, "hybrid_weight": hybrid, "score": round(score, 4)})

    for r in results:
        print(json.dumps(r))
    # Pick the setting with the best mean over English and French (quiet presets always filtered).
    by_setting = defaultdict(list)
    for r in results:
        if r["filter_quiet"]:
            by_setting[(r["csls_weight"], r.get("hybrid_weight"))].append(r["score"])
    (csls, hybrid), scores = max(by_setting.items(), key=lambda item: np.mean(item[1]))
    best = {"csls_weight": csls, "score_mean_en_fr": round(float(np.mean(scores)), 4)}
    if hybrid is not None:
        best["hybrid_weight"] = hybrid
    print("best:", json.dumps(best), f"({len(classes)} classes)")
    (args.data / "tuning.json").write_text(json.dumps({"classes": len(classes), "results": results, "best": best}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
