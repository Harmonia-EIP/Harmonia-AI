#!/usr/bin/env python3
"""Distill the CLAP text encoder (English) into a multilingual student (English + French).

The student sees each English sentence and its French translation and learns to output the
teacher's embedding of the English sentence (cosine loss + in-batch contrastive loss).

    python scripts/v2/train_text_encoder.py --corpus data/v2/corpus_en_fr.jsonl --out data/v2/text_student
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.v2.benchmark_set import held_out_sentences  # noqa: E402
from src.v2.clap_audio import TEACHER_ID, TEACHER_REVISION, ClapEmbedder, best_device  # noqa: E402
from src.v2.text_student import (  # noqa: E402
    MAX_TOKENS,
    STUDENT_BASE_ID,
    STUDENT_BASE_REVISION,
    TextStudent,
    load_student_tokenizer,
)


def teacher_embeddings(sentences, cache: Path) -> np.ndarray:
    """CLAP text embeddings of `sentences`, cached with the sentences so other scripts can reuse them."""
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        if data["sentences"].tolist() == list(sentences):
            return data["emb"]
    embedder = ClapEmbedder()
    emb = embedder.embed_text(sentences)
    np.savez(cache, emb=emb, sentences=np.array(sentences))
    del embedder
    return emb


@torch.no_grad()
def encode(model, tokenizer, sentences, device, batch=256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(sentences), batch):
        tok = tokenizer(sentences[i : i + batch], padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
        out.append(model(tok["input_ids"].to(device), tok["attention_mask"].to(device)).float().cpu().numpy())
    return np.concatenate(out)


def evaluate(model, tokenizer, val, teacher_val, device) -> dict:
    report = {}
    for lang in ("en", "fr"):
        student = encode(model, tokenizer, [r[lang] for r in val], device)
        cos = (student * teacher_val).sum(axis=1)
        sims = student @ teacher_val.T
        r1 = float((sims.argmax(axis=1) == np.arange(len(val))).mean())
        report[f"{lang}_cos_to_teacher"] = round(float(cos.mean()), 4)
        report[f"{lang}_retrieval_r1"] = round(r1, 4)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=BASE_DIR / "data" / "v2" / "corpus_en_fr.jsonl")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "data" / "v2" / "text_student")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--val-size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = best_device()

    rows = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    held_out = held_out_sentences()
    rows = [r for r in rows if r.get("fr") and r["en"].lower() not in held_out and r["fr"].lower() not in held_out]
    random.shuffle(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    teacher = teacher_embeddings([r["en"] for r in rows], args.out.parent / "teacher_text.npz")
    val, train = rows[: args.val_size], rows[args.val_size :]
    teacher_val, teacher_train = teacher[: args.val_size], teacher[args.val_size :]

    tokenizer = load_student_tokenizer()
    model = TextStudent().to(device)
    model.encoder.embeddings.word_embeddings.weight.requires_grad_(False)  # keep the 50-language vocabulary aligned
    print("baseline", json.dumps(evaluate(model, tokenizer, val, teacher_val, device)), flush=True)

    # Each English sentence appears twice per epoch: once in English, once in French.
    samples = [(r["en"], i) for i, r in enumerate(train)] + [(r["fr"], i) for i, r in enumerate(train)]
    steps_per_epoch = math.ceil(len(samples) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": args.lr}, {"params": model.proj.parameters(), "lr": args.head_lr}],
        weight_decay=0.01,
    )
    warmup = max(1, total_steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup) * max(0.0, (total_steps - step) / max(1, total_steps - warmup)),
    )
    teacher_t = torch.from_numpy(teacher_train).float()

    history = []
    step = 0
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(samples)
        running = 0.0
        for b in range(steps_per_epoch):
            batch = samples[b * args.batch : (b + 1) * args.batch]
            texts = [t for t, _ in batch]
            target = teacher_t[[i for _, i in batch]].to(device)
            tok = tokenizer(texts, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
            pred = model(tok["input_ids"].to(device), tok["attention_mask"].to(device))
            cos_loss = (1.0 - (pred * target).sum(dim=-1)).mean()
            logits = pred @ target.T / args.temperature
            labels = torch.arange(len(batch), device=device)
            loss = cos_loss + args.contrastive_weight * F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            running += float(loss)
            if step % 100 == 0:
                print(f"epoch {epoch + 1} step {step}/{total_steps} loss {running / 100:.4f} | {time.time() - start:.0f}s", flush=True)
                running = 0.0
        report = {"epoch": epoch + 1, **evaluate(model, tokenizer, val, teacher_val, device)}
        history.append(report)
        print("eval", json.dumps(report), flush=True)

    model.encoder.save_pretrained(args.out / "encoder")
    tokenizer.save_pretrained(args.out / "encoder")
    torch.save(model.proj.state_dict(), args.out / "proj.pt")
    meta = {
        "base_model": {"id": STUDENT_BASE_ID, "revision": STUDENT_BASE_REVISION},
        "teacher": {"id": TEACHER_ID, "revision": TEACHER_REVISION},
        "train_sentences": len(train),
        "val_sentences": len(val),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "history": history,
        "train_seconds": round(time.time() - start, 1),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
