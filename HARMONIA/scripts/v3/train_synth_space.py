#!/usr/bin/env python3
"""Train the synth space (src/v3/synth_space.py) and measure how well Harmonia understands synth words.

Training pairs are human-made only: bank presets with their own names / categories / comments, FSD50K
recordings with their titles and class labels; French versions come from the opus-mt translations of those
texts. 10 % of the presets and FSD50K's validation split are held out for the report:

- audio type accuracy: is a held-out preset's sound recognised as its labelled type (bass, pad, ...)?
- text -> type precision@20: for "pad", "soft pad", "nappe douce"..., are the 20 closest held-out presets
  of that type?
- held-out preset name -> its own sound: recall@10.

Each figure is compared with v3.0 (multilingual student -> prior -> CLAP audio space).

    python scripts/v3/train_synth_space.py      # -> data/v3/synth_space/{synth_space.pt,report.json}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.v2.build_text_corpus import clean_title, label_phrase  # noqa: E402
from src.presets.labels import expand, preset_text  # noqa: E402
from src.v2.clap_audio import best_device  # noqa: E402
from src.v2.text_student import encode_texts, load_trained_student  # noqa: E402
from src.v3 import vocabulary as V  # noqa: E402
from src.v3.prior import Prior  # noqa: E402
from src.v3.synth_space import SynthSpace, contrastive_loss, type_loss  # noqa: E402

BANK = BASE_DIR / "data" / "v3" / "bank"
STUDENT = BASE_DIR / "data" / "v3" / "text_student"
CORPUS = BASE_DIR / "data" / "v3" / "corpus_en_fr.jsonl"
FSD50K_DEV = BASE_DIR.parents[1] / "ai-v2" / "HARMONIA" / "data" / "v2" / "fsd50k_dev.npz"
OUT = BASE_DIR / "data" / "v3" / "synth_space"
ANALOG_WEIGHT = 4.0  # 2,117 analog presets next to 27,414 DX7 voices
QUERY_TEMPLATES = ("{}", "soft {}", "synth {}", "dark {}", "bright {}")
TYPE_WEIGHT = 0.3  # weight of the type agreement; 0.1 to 1.0 all help on the held-out presets


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def held_out(key: str, percent: int = 10) -> bool:
    return int(hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest()[:8], 16) % 100 < percent


def preset_segments(entry: dict) -> list:
    """The preset's own texts: full line, each readable name, its category, its comment."""
    out = [preset_text(entry)]
    for name in [str(entry.get("name", "")), *entry.get("aliases", [])[:4]]:
        out.append(expand(name))
    category = str(entry.get("category") or "").replace("\\", " ").replace("/", " ").replace("_", " ")
    out.append(expand(category))
    out.append(str(entry.get("comment") or "").strip())
    seen, kept = set(), []
    for text in out:
        if text and text not in seen:
            seen.add(text)
            kept.append(text)
    return kept


def translate(texts: list, cache_path: Path) -> dict:
    """English -> French with opus-mt (same model and jargon fixes as the v2/v3 corpora), cached."""
    cache = {}
    if cache_path.exists():
        cache = {r["en"]: r["fr"] for r in map(json.loads, cache_path.read_text(encoding="utf-8").splitlines())}
    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    if todo:
        from transformers import MarianMTModel, MarianTokenizer

        from scripts.v2.translate_corpus import TRANSLATOR_ID, TRANSLATOR_REVISION, fix_jargon

        tok = MarianTokenizer.from_pretrained(TRANSLATOR_ID, revision=TRANSLATOR_REVISION)  # nosec B615
        model = MarianMTModel.from_pretrained(TRANSLATOR_ID, revision=TRANSLATOR_REVISION).eval()  # nosec B615
        print(f"translating {len(todo)} texts", flush=True)
        with open(cache_path, "a", encoding="utf-8") as f, torch.no_grad():
            for i in range(0, len(todo), 64):
                chunk = todo[i:i + 64]
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=48)
                out = model.generate(**enc, num_beams=2, max_new_tokens=min(64, enc["input_ids"].shape[1] * 2 + 8),
                                     no_repeat_ngram_size=3)
                for en, fr in zip(chunk, tok.batch_decode(out, skip_special_tokens=True)):
                    fr = fix_jargon(en, fr)
                    cache[en] = fr
                    f.write(json.dumps({"en": en, "fr": fr}, ensure_ascii=False) + "\n")
    return {t: cache[t] for t in texts}


class Data:
    """Every training/evaluation array, with texts deduplicated and embedded once."""

    def __init__(self, embed):
        z = np.load(BANK / "bank.npz")
        self.entries = [json.loads(line) for line in open(BANK / "entries.jsonl", encoding="utf-8")]
        self.bank_emb = normalize(z["emb"].astype(np.float32))  # (n, notes, 512)
        self.kind = z["kind"]
        self.bank_types = np.stack([V.multi_hot(V.preset_types(e)) for e in self.entries])
        self.bank_test = np.array([held_out(e["id"]) for e in self.entries])

        fsd = np.load(FSD50K_DEV)
        self.fsd_emb = normalize(fsd["emb"].astype(np.float32))
        self.fsd_types = np.stack([V.multi_hot(V.fsd_types(str(lab))) for lab in fsd["labels"]])
        self.fsd_test = fsd["split"] == "val"

        known_fr = {}
        for row in map(json.loads, CORPUS.read_text(encoding="utf-8").splitlines()):
            if row.get("fr") and row["fr"] != row["en"]:
                known_fr[row["en"]] = row["fr"]
        bank_en = [preset_segments(e) for e in self.entries]
        fsd_en = []
        for title, labels in zip(fsd["title"], fsd["labels"]):
            texts = [label_phrase(str(labels).split(",")[0])]
            title = clean_title(str(title))
            if len(title.split()) >= 2:
                texts.append(title)
            fsd_en.append(texts)
        # French: corpus translations, plus fresh ones for short texts naming a type (categories like "pads")
        typed = [t for texts in bank_en for t in texts if V.words_to_types(t) and t not in known_fr and len(t) < 60]
        OUT.mkdir(parents=True, exist_ok=True)
        fresh = translate(sorted(set(typed)), OUT / "fr_cache.jsonl")
        fr = {**known_fr, **{k: v for k, v in fresh.items() if v and v != k}}

        texts, group = [], []
        index = {}

        def add(text: str, group_of: str) -> int:
            if text not in index:
                index[text] = len(texts)
                texts.append(text)
                group.append(group_of)
            return index[text]

        self.bank_text_ids = [[add(t, t) for t in seg] + [add(fr[t], t) for t in seg if t in fr] for seg in bank_en]
        self.fsd_text_ids = [[add(t, t) for t in seg] + [add(fr[t], t) for t in seg if t in fr] for seg in fsd_en]
        self.texts = texts
        groups = {g: i for i, g in enumerate(dict.fromkeys(group))}
        self.text_group = np.array([groups[g] for g in group])
        self.text_types = np.stack([V.multi_hot(V.words_to_types(g)) for g in group])  # types of the English source
        print(f"{len(texts)} texts ({len(fr)} with French), embedding...", flush=True)
        self.text_emb = normalize(embed(texts).astype(np.float32))


def train(data: Data, steps: int, device, seed: int) -> SynthSpace:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SynthSpace().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=steps, pct_start=0.05)
    bank_train = np.array([i for i in np.flatnonzero(~data.bank_test) if data.bank_text_ids[i]])
    weights = np.where(data.kind[bank_train] == "analog", ANALOG_WEIGHT, 1.0)
    weights /= weights.sum()
    fsd_train = np.flatnonzero(~data.fsd_test)
    text_emb = torch.from_numpy(data.text_emb).to(device)
    text_types = torch.from_numpy(data.text_types).to(device)
    start, running = time.time(), None
    for step in range(1, steps + 1):
        b = rng.choice(bank_train, 384, p=weights)
        f = rng.choice(fsd_train, 128)
        notes = rng.integers(0, data.bank_emb.shape[1], len(b))
        audio = np.concatenate([data.bank_emb[b, notes], data.fsd_emb[f]])
        audio_types = np.concatenate([data.bank_types[b], data.fsd_types[f]])
        tid = np.array([rng.choice(data.bank_text_ids[i]) for i in b] + [rng.choice(data.fsd_text_ids[i]) for i in f])
        a = torch.from_numpy(audio).to(device)
        a = a + 0.02 * torch.randn_like(a)
        ids = torch.from_numpy(tid).to(device)
        az = model.audio(a)
        tz = model.text(text_emb[ids])
        groups = torch.from_numpy(data.text_group[tid]).to(device)
        loss = (contrastive_loss(tz, az, groups, model.log_scale.exp().clamp(max=100))
                + 0.5 * type_loss(model.type_logits(az), torch.from_numpy(audio_types).to(device))
                + 0.5 * type_loss(model.type_logits(tz), text_types[ids]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        if step % 1000 == 0 or step == steps:
            print(f"step {step}/{steps} loss {running:.4f} ({time.time() - start:.0f}s)", flush=True)
    return model.cpu().eval()


def single_type(types: np.ndarray) -> np.ndarray:
    return np.where(types.sum(1) == 1, types.argmax(1), -1)


def evaluate(audio_space, text_space, data: Data, queries: dict, typed=None) -> dict:
    """audio_space(emb (n, 512)) and text_space(texts) give vectors of one shared space (normalised).

    typed = (audio_type_logp, text_type_p): the closeness is then helped by the type each side recognises,
    score = cosine + TYPE_WEIGHT * mean log P(prompt's types | sound)."""
    test = np.flatnonzero(data.bank_test)
    ref = np.flatnonzero(~data.bank_test)
    a_test = audio_space(normalize(data.bank_emb[test].mean(1)))
    a_ref = audio_space(normalize(data.bank_emb[ref].mean(1)))
    y_test, y_ref = single_type(data.bank_types[test]), single_type(data.bank_types[ref])
    report = {}
    keep = y_test >= 0
    keep_ref = y_ref >= 0
    sims = a_test[keep] @ a_ref[keep_ref].T  # neighbours by sound only
    nn = np.argsort(-sims, axis=1)[:, :5]
    votes = np.array([np.bincount(row, minlength=len(V.TYPES)).argmax() for row in y_ref[keep_ref][nn]])
    report["audio_type_5nn"] = round(float(np.mean(votes == y_test[keep])), 3)
    analog = data.kind[test][keep] == "analog"
    report["audio_type_5nn_analog"] = round(float(np.mean((votes == y_test[keep])[analog])), 3)
    # text -> type precision@20 over held-out single-type presets
    pool = a_test[keep]
    pool_y = y_test[keep]
    pool_logp = typed[0](a_test[keep]) if typed else None
    for name, (phrases, types) in queries.items():
        q = text_space(phrases)
        scores = q @ pool.T
        if typed:
            p = typed[1](q)
            scores = scores + TYPE_WEIGHT * (p @ pool_logp.T) / np.maximum(p.sum(1, keepdims=True), 1e-6)
        top = np.argsort(-scores, axis=1)[:, :20]
        prec = [float(np.mean(pool_y[row] == V.TYPES.index(t))) for row, t in zip(top, types)]
        report[f"text_type_p@20_{name}"] = round(float(np.mean(prec)), 3)
        report[f"text_type_p@20_{name}_by_type"] = {t: round(p, 2) for t, p in zip(types, prec)}
    # held-out preset name (with a type word) -> its own sound among held-out presets
    named = [i for i, e in zip(test, (data.entries[j] for j in test)) if V.words_to_types(preset_text(e))]
    pos = {j: k for k, j in enumerate(test)}
    q = text_space([preset_text(data.entries[i]) for i in named])
    ranks = np.argsort(-(q @ a_test.T), axis=1)[:, :10]
    report["name_to_sound_recall@10"] = round(float(np.mean([pos[i] in row for i, row in zip(named, ranks)])), 3)
    report["name_to_sound_pool"] = int(len(test))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = best_device()
    student, tokenizer = load_trained_student(STUDENT, device)

    def embed(texts):
        return encode_texts(student, tokenizer, texts, device)

    data = Data(embed)
    en_queries = [tpl.format(t) for tpl in QUERY_TEMPLATES for t in V.TYPES]
    types = [t for _ in QUERY_TEMPLATES for t in V.TYPES]
    fr_queries = translate(en_queries, OUT / "fr_cache.jsonl")
    queries = {"en_word": (en_queries[:len(V.TYPES)], V.TYPES),
               "en_adjective": (en_queries[len(V.TYPES):], types[len(V.TYPES):]),
               "fr_word": ([fr_queries[q] for q in en_queries[:len(V.TYPES)]], V.TYPES),
               "fr_adjective": ([fr_queries[q] for q in en_queries[len(V.TYPES):]], types[len(V.TYPES):])}

    # v3.0: student -> prior -> CLAP audio space
    prior = Prior()
    prior.load_state_dict(torch.load(BASE_DIR / "data" / "v3" / "generator" / "prior.pt", map_location="cpu",
                                     weights_only=True))
    prior.eval()

    def prior_text(texts):
        with torch.no_grad():
            return prior(torch.from_numpy(normalize(embed(texts).astype(np.float32)))).numpy()

    before = evaluate(lambda a: a, prior_text, data, queries)
    print("v3.0", json.dumps(before, ensure_ascii=False), flush=True)

    model = train(data, args.steps, device, args.seed)

    def space_audio(a):
        with torch.no_grad():
            return model.audio(torch.from_numpy(a)).numpy()

    def space_text(texts):
        with torch.no_grad():
            return model.text(torch.from_numpy(normalize(embed(texts).astype(np.float32)))).numpy()

    def audio_logp(z):
        with torch.no_grad():
            return torch.log_softmax(model.type_logits(torch.from_numpy(z)), -1).numpy()

    def text_p(z):
        with torch.no_grad():
            return torch.sigmoid(model.type_logits(torch.from_numpy(z))).numpy()

    after = evaluate(space_audio, space_text, data, queries)
    print("synth space", json.dumps(after, ensure_ascii=False), flush=True)
    typed = evaluate(space_audio, space_text, data, queries, typed=(audio_logp, text_p))
    print("synth space + types", json.dumps(typed, ensure_ascii=False), flush=True)
    torch.save(model.state_dict(), OUT / "synth_space.pt")
    # conditions for the generators: each bank preset's sound and each of its own texts, in the synth space
    n, notes = data.bank_emb.shape[:2]
    audio_z = space_audio(data.bank_emb.reshape(n * notes, -1)).reshape(n, notes, -1)
    owner = np.array([i for i, ids in enumerate(data.bank_text_ids) for _ in ids])
    text_ids = np.array([t for ids in data.bank_text_ids for t in ids])
    with torch.no_grad():
        text_z = model.text(torch.from_numpy(data.text_emb[text_ids])).numpy()
        text_types = torch.sigmoid(model.type_logits(torch.from_numpy(text_z))).numpy()
        audio_types = torch.sigmoid(model.type_logits(torch.from_numpy(audio_z))).numpy()
    np.savez_compressed(OUT / "bank_cond.npz", audio_z=audio_z.astype(np.float16), text_z=text_z.astype(np.float16),
                        text_owner=owner, test=data.bank_test, labels=data.bank_types,
                        audio_types=audio_types.astype(np.float16), text_types=text_types.astype(np.float16))
    report = {"v3.0": before, "synth_space": after, "synth_space_typed": typed, "type_weight": TYPE_WEIGHT, "queries_fr": queries["fr_word"][0] + queries["fr_adjective"][0],
              "texts": len(data.texts), "args": vars(args)}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
