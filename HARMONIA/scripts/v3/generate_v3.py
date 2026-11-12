#!/usr/bin/env python3
"""Generate presets from text with the v3 models, next to the "selection" baseline (offline evaluation).

For each prompt: the multilingual encoder puts it in CLAP's text space, the prior carries it to the sound
space, each generator (DX7, analog) draws candidates for that target, Harmonia plays them and CLAP listens;
the best candidate wins. The baseline picks the bank preset whose sound is closest to the same target.

    python scripts/v3/generate_v3.py --prompts my_prompts.json --out data/v3/generations
    (my_prompts.json: ["piano électrique doux", ...])
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.presets import dx7_render  # noqa: E402
from src.synth.engine_v3 import render_batch  # noqa: E402
from src.v2.clap_audio import best_device  # noqa: E402
from src.v2.text_student import encode_texts, load_trained_student  # noqa: E402
from src.v3 import codecs  # noqa: E402
from src.v3.diffusion import Diffusion  # noqa: E402
from src.v3.prior import Prior  # noqa: E402

GEN = BASE_DIR / "data" / "v3" / "generator"
BANK = BASE_DIR / "data" / "v3" / "bank"
STUDENT = BASE_DIR / "data" / "v3" / "text_student"
NOTES = (45, 69)
SR = 48000


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


class Generator:
    def __init__(self, device):
        from src.v2.clap_audio import ClapEmbedder

        self.device = device
        self.student, self.tokenizer = load_trained_student(STUDENT, device)
        self.prior = Prior()
        self.prior.load_state_dict(torch.load(GEN / "prior.pt", map_location="cpu", weights_only=True))
        self.prior.eval()
        self.models = {}
        for kind, dim in (("dx7", codecs.DX7_DIM), ("analog", codecs.ANALOG_DIM)):
            diffusion = Diffusion.create(dim)
            diffusion.model.load_state_dict(torch.load(GEN / f"{kind}.pt", map_location="cpu", weights_only=True))
            diffusion.model.to(device).eval()
            self.models[kind] = diffusion
        self.clap = ClapEmbedder(half=True)
        z = np.load(BANK / "bank.npz")
        self.bank_emb = normalize(normalize(z["emb"].astype(np.float32)).mean(axis=1))
        self.bank_kind = z["kind"]
        self.bank_entries = [json.loads(line) for line in open(BANK / "entries.jsonl", encoding="utf-8")]

    def target(self, prompts):
        text = encode_texts(self.student, self.tokenizer, prompts, self.device)
        with torch.no_grad():
            return text, self.prior(torch.from_numpy(text)).numpy()

    def render(self, kind: str, vector: np.ndarray) -> np.ndarray:
        if kind == "dx7":
            voice = codecs.dx7_decode(vector)
            return np.stack([dx7_render.render(voice, note=n, total_seconds=4.0) for n in NOTES])
        physical = codecs.analog_decode(vector)[None, :]
        return render_batch(physical, np.array(NOTES), 100 / 127, 1.5, 4.0, SR, 7)[0]

    def listen(self, audio: np.ndarray) -> np.ndarray:
        """Mean CLAP embedding over the notes of each candidate: audio is (n, notes, samples)."""
        flat = [np.nan_to_num(a) for a in audio.reshape(-1, audio.shape[-1])]
        emb = normalize(self.clap.embed_audio(flat, batch_size=32)).reshape(audio.shape[0], audio.shape[1], -1)
        return normalize(emb.mean(axis=1))

    def generate(self, target: np.ndarray, candidates: int, guidance: float, seed: int):
        results = []
        for kind, diffusion in self.models.items():
            gen = torch.Generator(device="cpu").manual_seed(seed)
            cond = torch.from_numpy(np.repeat(target[None, :], candidates, axis=0)).float()
            x = diffusion.sample(cond.to(self.device), guidance=guidance,
                                 generator=None if self.device.type != "cpu" else gen).cpu().numpy()
            audio = np.stack([self.render(kind, v) for v in x])
            emb = self.listen(audio)
            scores = emb @ target
            for v, s, a in zip(x, scores, audio):
                results.append({"kind": kind, "vector": v, "score": float(s), "audio": a})
        return sorted(results, key=lambda r: -r["score"])

    def select(self, target: np.ndarray, k: int = 1):
        scores = self.bank_emb @ target
        best = np.argsort(-scores)[:k]
        return [(self.bank_entries[i], float(scores[i])) for i in best]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=BASE_DIR / "data" / "v3" / "generations")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    device = best_device()
    gen = Generator(device)
    text_emb, targets = gen.target(prompts)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (prompt, target) in enumerate(zip(prompts, targets)):
        ranked = gen.generate(target, args.candidates, args.guidance, args.seed + i)
        best = ranked[0]
        chosen, chosen_score = gen.select(target)[0]
        np.savez_compressed(args.out / f"prompt_{i:03d}.npz", generated=best["audio"].astype(np.float32))
        rows.append({
            "prompt": prompt, "generated_kind": best["kind"], "generated_score": best["score"],
            "generated_vector": best["vector"].tolist(),
            "candidate_scores": [round(r["score"], 4) for r in ranked],
            "selected": {"id": chosen["id"], "name": chosen["name"], "kind": chosen["kind"], "score": chosen_score},
            "text_target_cos": float(normalize(text_emb[i]) @ target),
        })
        print(f"{prompt[:40]:40s} generated {best['kind']:6s} {best['score']:.3f} | selected "
              f"{chosen['name'][:16]:16s} ({chosen['kind']}) {chosen_score:.3f}", flush=True)
    (args.out / "generations.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
