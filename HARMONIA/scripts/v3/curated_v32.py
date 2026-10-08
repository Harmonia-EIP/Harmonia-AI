#!/usr/bin/env python3
"""v3.2 test: a curated library of professional presets inside Harmonia, chosen by how they sound.

Blind round 2 showed the synth space picks presets whose *names* match the prompt (GlassBreak, MOOG BASS,
Tabla 2...), mostly amateur DX7 voices that sound poor, while v3.0's choice by *sound* (CLAP) was rated
better. Here:
- library: presets made by professional designers only: Yamaha's own DX7 cartridges (ROM1-4, VRC, TX816,
  DX5, DX7II...) and the OB-Xf / Surge presets Harmonia reproduces faithfully (bank threshold), minus those
  Malo rated 2 or less in listening round 3; ~5,000 presets, a few MB with their embeddings;
- choice: the v3.0 sound target (text -> prior -> CLAP sound), restricted to the type the prompt names
  (synth space) when it names one clearly;
- "varied": the chosen preset changed lightly by the v3.1 generator (15 % of the noise schedule).

For each prompt three sounds go to a blind page: v3.0's choice in the whole bank (reference), the curated
choice and its light variation.

    python scripts/v3/curated_v32.py --prompts benchmarks/v3_prompts_p1.json --out data/v3/generations_v32
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.v3.generate_v31 import SynthGenerator, normalize, render  # noqa: E402
from src.v2.clap_audio import best_device  # noqa: E402
from src.v2.text_student import encode_texts  # noqa: E402
from src.v3.prior import Prior  # noqa: E402
from src.v3.vocabulary import TYPES, preset_types  # noqa: E402

DATA = BASE_DIR / "data" / "v3"
RATINGS = DATA / "listening" / "ratings_r3"
TYPE_CLEAR = 0.5  # the prompt names a type when the synth space is this sure of it
TYPE_KEEP = 0.2  # a preset keeps that type when its sound is recognised as it with this probability (or labelled)
LIGHT = 0.15


def curated_mask(entries: list, low_rated: set) -> np.ndarray:
    dx7 = {}
    for line in open(DATA / "presets" / "dx7.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        dx7[rec["id"]] = any("Original Yamaha" in p for p in rec["paths"])
    keep = []
    for e in entries:
        pro = dx7.get(e["id"], False) if e["kind"] == "dx7" else True
        keep.append(pro and e["id"] not in low_rated)
    return np.array(keep)


def low_rated_ids(path: Path) -> set:
    """Bank ids of presets rated 2/5 or less in listening round 3 (ratings saved from the page's database)."""
    out = set()
    for f in glob.glob(str(path / "*-r3.json")):
        doc = json.loads(Path(f).read_text(encoding="utf-8"))
        if doc.get("score") and doc["score"] <= 2:
            source, key = Path(f).stem[:-3].split("-", 1)
            out.add(f"{source}:{key}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DATA / "generations_v32")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = best_device()
    gen = SynthGenerator(device, candidates=8, strength=LIGHT)
    prior = Prior()
    prior.load_state_dict(torch.load(DATA / "generator" / "prior.pt", map_location="cpu", weights_only=True))
    prior.eval()
    z = np.load(DATA / "bank" / "bank.npz")
    clap_bank = normalize(normalize(z["emb"].astype(np.float32)).mean(axis=1))
    low = low_rated_ids(RATINGS)
    library = curated_mask(gen.entries, low)
    labels = [preset_types(e) for e in gen.entries]
    bank_types = np.exp(gen.bank_logp)
    print(f"library: {library.sum()} presets ({(library & (gen.bank_kind == 'dx7')).sum()} Yamaha DX7, "
          f"{(library & (gen.bank_kind == 'analog')).sum()} analog); {len(low)} low-rated left out", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "library_ids.npy", np.array([e["id"] for e, k in zip(gen.entries, library) if k]))

    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    text = normalize(encode_texts(gen.student, gen.tokenizer, prompts, device).astype(np.float32))
    with torch.no_grad():
        targets = prior(torch.from_numpy(text)).numpy()
    conds, types = gen.text(prompts)
    rows = []
    for i, (prompt, target, c, t) in enumerate(zip(prompts, targets, conds, types)):
        sound = clap_bank @ target
        reference = int(np.argmax(sound))  # v3.0 selection
        allowed = library.copy()
        top = int(np.argmax(t))
        if t[top] >= TYPE_CLEAR:
            allowed &= np.array([bank_types[k, top] >= TYPE_KEEP or TYPES[top] in labels[k] for k in range(len(labels))])
        chosen = int(np.argmax(np.where(allowed, sound, -np.inf)))
        kind = str(gen.bank_kind[chosen])
        anchor = gen.bank_vector(chosen)
        x = gen.draw(kind, c, t, args.seed + i, anchors=[anchor] * 8)
        audio = np.stack([render(kind, v) for v in x])
        flat = [np.nan_to_num(a) for a in audio.reshape(-1, audio.shape[-1])]
        emb = normalize(normalize(gen.clap.embed_audio(flat, batch_size=32)).reshape(len(x), audio.shape[1], -1).mean(1))
        peak = np.abs(audio).reshape(len(x), -1).max(axis=1)
        score = np.where(peak > 1e-3, emb @ target, -np.inf)
        best = int(np.argmax(score))
        row = {"prompt": prompt, "type": TYPES[top] if t[top] >= TYPE_CLEAR else None,
               "reference": {"id": gen.entries[reference]["id"], "name": gen.entries[reference]["name"],
                             "kind": str(gen.bank_kind[reference]), "score": float(sound[reference])},
               "curated": {"id": gen.entries[chosen]["id"], "name": gen.entries[chosen]["name"], "kind": kind,
                           "score": float(sound[chosen])},
               "varied": {"kind": kind, "vector": x[best].tolist(), "score": float(score[best])}}
        rows.append(row)
        print(f"{prompt[:34]:34s} {str(row['type']):6s} ref {row['reference']['name'][:16]:16s} "
              f"{row['reference']['score']:.2f} | curated {row['curated']['name'][:16]:16s} ({kind}) "
              f"{row['curated']['score']:.2f} | varied {row['varied']['score']:.2f}", flush=True)
    (args.out / "generations.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
