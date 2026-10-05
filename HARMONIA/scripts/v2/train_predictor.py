#!/usr/bin/env python3
"""Train the v2 parameter predictor: student text embedding -> charter parameters.

Training pairs (inputs are always *student* text embeddings, as at inference time):
- corpus: each sentence (English and its French translation) -> the bank preset whose audio is
  closest to the CLAP teacher's embedding of the English sentence;
- fsd50k: label phrase and cleaned title of each real recording -> the bank preset whose audio
  is closest to that recording (audio-to-audio), i.e. a preset that imitates it.

    python scripts/v2/train_predictor.py --data data/v2 --out data/v2/predictor
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.v2.build_text_corpus import clean_title, label_phrase  # noqa: E402
from src.charter import DISCRETE_INDICES, DISCRETE_STEPS, PARAM_NAMES  # noqa: E402
from src.v2.benchmark_set import held_out_sentences  # noqa: E402
from src.v2.clap_audio import best_device  # noqa: E402
from src.v2.predictor import ParamPredictor  # noqa: E402
from src.v2.retrieval import AUDIBLE_DB_FLOOR, top_indices  # noqa: E402
from src.v2.text_student import encode_texts, load_trained_student  # noqa: E402

# Same perceptual weighting as scripts/train.py (CHARTER_LOSS_WEIGHTS).
LOSS_WEIGHTS = {
    "osc_1_waveform": 2.0, "osc_2_waveform": 1.5, "osc_mix": 1.0, "osc_2_detune": 0.8, "noise_level": 0.8,
    "filter_cutoff": 2.0, "filter_resonance": 1.2, "filter_type": 2.0, "amp_attack": 1.5, "amp_decay": 1.2,
    "amp_sustain": 1.2, "amp_release": 1.5, "filter_env_amount": 1.0, "filter_env_decay": 1.0, "lfo_rate": 0.8,
    "lfo_to_pitch": 0.6, "lfo_to_cutoff": 0.8, "velocity_to_filter": 0.8, "distortion_mix": 1.5, "reverb_mix": 1.2,
}


def load_bank(path: Path):
    bank = np.load(path)
    emb = bank["emb"].astype(np.float32).mean(axis=1)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return bank["params"].astype(np.float32), emb


def load_priors(data: Path):
    """(usable bank indices, hub scores, tuned retrieval settings) from analyze_bank.py / tune_retrieval.py."""
    extra = np.load(data / "bank" / "bank_extra.npz")
    usable = np.flatnonzero(extra["audible_db"] >= AUDIBLE_DB_FLOOR)
    tuning_path = data / "tuning.json"
    best = json.loads(tuning_path.read_text(encoding="utf-8"))["best"] if tuning_path.exists() else {}
    return usable, extra["hub"].astype(np.float32), best


def nearest(queries: np.ndarray, bank_emb: np.ndarray, usable: np.ndarray, hub: np.ndarray, csls_weight: float) -> np.ndarray:
    """Index (in the full bank) of the best audible preset for each query."""
    return usable[top_indices(queries, bank_emb[usable], hub[usable], csls_weight, k=1)[:, 0]]


def discrete_targets(params: torch.Tensor):
    targets = []
    for idx in DISCRETE_INDICES:
        n = len(DISCRETE_STEPS[idx])
        targets.append(torch.round(params[:, idx] * (n - 1)).long().clamp(0, n - 1))
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "v2")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "data" / "v2" / "predictor")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = best_device()
    args.out.mkdir(parents=True, exist_ok=True)

    bank_params, bank_emb = load_bank(args.data / "bank" / "bank.npz")
    usable, hub, tuned = load_priors(args.data)
    csls = float(tuned.get("csls_weight", 0.5))
    print(f"{len(usable)}/{len(bank_params)} audible presets, csls_weight={csls}", flush=True)
    student, tokenizer = load_trained_student(args.data / "text_student", device)

    # Corpus pairs: targets from teacher(EN) -> bank retrieval, inputs = student(EN) and student(FR).
    teacher = np.load(args.data / "teacher_text.npz")
    sentences = teacher["sentences"].tolist()
    targets = nearest(teacher["emb"].astype(np.float32), bank_emb, usable, hub, csls)
    fr_by_en = {}
    for line in (args.data / "corpus_en_fr.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("fr"):
            fr_by_en[row["en"]] = row["fr"]
    texts = list(sentences) + [fr_by_en[s] for s in sentences if s in fr_by_en]
    text_targets = list(targets) + [t for s, t in zip(sentences, targets) if s in fr_by_en]
    sources = ["corpus_en"] * len(sentences) + ["corpus_fr"] * (len(texts) - len(sentences))

    # FSD50K pairs: real recording -> closest bank preset (audio to audio); text = its label and title.
    fsd = np.load(args.data / "fsd50k_dev.npz")
    fsd_targets = nearest(fsd["emb"].astype(np.float32), bank_emb, usable, hub, csls)
    for labels, title, target in zip(fsd["labels"], fsd["title"], fsd_targets):
        leaf = label_phrase(str(labels).split(",")[0])
        texts.append(leaf)
        text_targets.append(target)
        sources.append("fsd50k_label")
        cleaned = clean_title(str(title))
        if len(cleaned.split()) >= 2:
            texts.append(cleaned)
            text_targets.append(target)
            sources.append("fsd50k_title")

    held_out = held_out_sentences()
    keep = [i for i, t in enumerate(texts) if t.strip().lower() not in held_out]
    texts = [texts[i] for i in keep]
    text_targets = [text_targets[i] for i in keep]
    sources = [sources[i] for i in keep]
    print(f"{len(texts)} training texts ({ {s: sources.count(s) for s in set(sources)} })", flush=True)
    x = encode_texts(student, tokenizer, texts, device)
    y = bank_params[np.array(text_targets)]
    del student

    order = rng.permutation(len(x))
    n_val = int(len(x) * args.val_frac)
    val_idx, train_idx = order[:n_val], order[n_val:]
    x_t = torch.from_numpy(x).float()
    y_t = torch.from_numpy(y).float()
    weights = torch.tensor([LOSS_WEIGHTS[n] for n in PARAM_NAMES], dtype=torch.float32, device=device)
    cont_mask = torch.ones(len(PARAM_NAMES), dtype=torch.bool, device=device)
    cont_mask[list(DISCRETE_INDICES)] = False

    model = ParamPredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def run_loss(xb, yb):
        values, logits = model(xb)
        mse = ((values - yb) ** 2 * weights)[:, cont_mask].mean()
        ce = sum(F.cross_entropy(lg, tg) * weights[idx] for lg, tg, idx in zip(logits, discrete_targets(yb), DISCRETE_INDICES))
        return mse + 0.1 * ce / len(DISCRETE_INDICES)

    history = []
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = train_idx[rng.permutation(len(train_idx))]
        total = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i : i + args.batch]
            xb, yb = x_t[b].to(device), y_t[b].to(device)
            xb = F.normalize(xb + 0.02 * torch.randn_like(xb), dim=-1)  # small embedding noise for robustness
            loss = run_loss(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(b)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(run_loss(x_t[val_idx].to(device), y_t[val_idx].to(device)))
            values, _ = model(x_t[val_idx].to(device))
            mae = (values.cpu() - y_t[val_idx]).abs().mean(dim=0).numpy()
        report = {"epoch": epoch + 1, "train_loss": round(total / len(train_idx), 5), "val_loss": round(val_loss, 5), "val_mae": round(float(mae.mean()), 4)}
        history.append(report)
        print(json.dumps(report), flush=True)

    torch.save(model.state_dict(), args.out / "predictor.pt")
    meta = {
        "pairs": len(texts),
        "sources": {s: sources.count(s) for s in sorted(set(sources))},
        "epochs": args.epochs,
        "csls_weight": csls,
        "audible_presets": int(len(usable)),
        "history": history,
        "val_mae_per_param": {n: round(float(v), 4) for n, v in zip(PARAM_NAMES, mae)},
        "train_seconds": round(time.time() - start, 1),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
