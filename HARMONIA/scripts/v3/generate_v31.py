#!/usr/bin/env python3
"""v3.1 text -> preset, with the synth space (scripts/v3/train_synth_space.py), for offline evaluation.

For each prompt, three answers:
- "generation": the synth-space generators draw DX7 voices and analog presets for the prompt itself;
- "variation": the same generators start from the real presets closest to the prompt, noised part way,
  and walk back towards the prompt (the presets' settings travel with the model);
- "selection": the real preset closest to the prompt (yardstick).

A sound matches a prompt when it is close to it in the synth space and is of the type the prompt names
(bass, pad, ...; scripts/v3/train_synth_space.py). Candidates are played and heard through CLAP + the synth
space. The kept one is not simply the best match (picking the best of many by one judge favours sounds that
fool the judge): it must also be typical of what the generator draws for that prompt. Both engines compete.

    python scripts/v3/generate_v31.py --prompts benchmarks/v3_prompts_p1.json --out data/v3/generations_p2
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
from scripts.v3.train_synth_space import TYPE_WEIGHT  # noqa: E402
from src.v3.synth_space import SynthSpace  # noqa: E402
from src.v3.vocabulary import TYPES  # noqa: E402

GEN = BASE_DIR / "data" / "v3" / "generator_synth"
SPACE = BASE_DIR / "data" / "v3" / "synth_space"
BANK = BASE_DIR / "data" / "v3" / "bank"
STUDENT = BASE_DIR / "data" / "v3" / "text_student"
NOTES = (45, 69)  # the bank's CLAP notes
SR = 48000
KINDS = ("dx7", "analog")
TYPICAL_WEIGHT = 0.5


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def decode(kind: str, vector: np.ndarray):
    return codecs.dx7_decode(vector) if kind == "dx7" else codecs.analog_decode(vector)


def render(kind: str, vector: np.ndarray) -> np.ndarray:
    if kind == "dx7":
        voice = codecs.dx7_decode(vector)
        return np.stack([dx7_render.render(voice, note=n, total_seconds=4.0) for n in NOTES])
    return render_batch(codecs.analog_decode(vector)[None, :], np.array(NOTES), 100 / 127, 1.5, 4.0, SR, 7)[0]


class SynthGenerator:
    def __init__(self, device, candidates: int = 8, guidance: float = 2.0, strength: float = 0.4):
        from src.v2.clap_audio import ClapEmbedder

        self.device, self.n, self.guidance, self.strength = device, candidates, guidance, strength
        self.student, self.tokenizer = load_trained_student(STUDENT, device)
        self.space = SynthSpace()
        self.space.load_state_dict(torch.load(SPACE / "synth_space.pt", map_location="cpu", weights_only=True))
        self.space.eval()
        self.models = {}
        for kind, dim in (("dx7", codecs.DX7_DIM), ("analog", codecs.ANALOG_DIM)):
            diffusion = Diffusion.create(dim, cond_dim=self.space.types.in_features + len(TYPES))
            diffusion.model.load_state_dict(torch.load(GEN / f"{kind}.pt", map_location="cpu", weights_only=True))
            diffusion.model.to(device).eval()
            self.models[kind] = diffusion
        self.clap = ClapEmbedder(half=True)
        z = np.load(BANK / "bank.npz")
        cond = np.load(SPACE / "bank_cond.npz")
        self.bank_z = normalize(normalize(cond["audio_z"].astype(np.float32)).mean(axis=1))
        self.bank_logp = self.type_logp(self.bank_z)
        self.bank_kind = z["kind"]
        self.bank_analog = z["analog"]
        self.bank_dx7 = z["dx7"]
        self.entries = [json.loads(line) for line in open(BANK / "entries.jsonl", encoding="utf-8")]

    def type_logp(self, z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.log_softmax(self.space.type_logits(torch.from_numpy(z)), -1).numpy()

    def text(self, prompts):
        """Synth-space vector of each prompt and the types it names (probabilities)."""
        emb = normalize(encode_texts(self.student, self.tokenizer, prompts, self.device).astype(np.float32))
        with torch.no_grad():
            c = self.space.text(torch.from_numpy(emb))
            types = torch.sigmoid(self.space.type_logits(c)).numpy()
        return c.numpy(), types

    @staticmethod
    def match(z: np.ndarray, logp: np.ndarray, c: np.ndarray, types: np.ndarray) -> np.ndarray:
        return z @ c + TYPE_WEIGHT * (logp @ types) / max(float(types.sum()), 1e-6)

    def hear(self, audio: np.ndarray) -> np.ndarray:
        """Synth-space vector of each candidate: audio is (n, notes, samples)."""
        flat = [np.nan_to_num(a) for a in audio.reshape(-1, audio.shape[-1])]
        emb = normalize(self.clap.embed_audio(flat, batch_size=32))
        with torch.no_grad():
            z = self.space.audio(torch.from_numpy(emb.astype(np.float32))).numpy()
        return normalize(z.reshape(audio.shape[0], audio.shape[1], -1).mean(axis=1))

    def nearest(self, c: np.ndarray, types: np.ndarray, kind: str | None = None, k: int = 1):
        scores = self.match(self.bank_z, self.bank_logp, c, types)
        if kind:
            scores = np.where(self.bank_kind == kind, scores, -np.inf)
        best = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in best]

    def bank_vector(self, i: int) -> np.ndarray:
        if self.bank_kind[i] == "dx7":
            return codecs.dx7_encode(dx7_render.from_patch(bytes(self.bank_dx7[i])))
        return codecs.analog_encode(self.bank_analog[i].astype(np.float64))

    def draw(self, kind: str, c: np.ndarray, types: np.ndarray, seed: int, anchors=None) -> np.ndarray:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        n = self.n if anchors is None else len(anchors)
        full = np.concatenate([c, types]).astype(np.float32)
        cond = torch.from_numpy(np.repeat(full[None, :], n, axis=0)).to(self.device)
        kwargs = {}
        if anchors is not None:
            kwargs = {"start": torch.from_numpy(np.stack(anchors)).float(), "strength": self.strength}
        noise_gen = gen if self.device.type == "cpu" else None
        return self.models[kind].sample(cond, guidance=self.guidance, generator=noise_gen, **kwargs).cpu().numpy()

    def answer(self, c: np.ndarray, types: np.ndarray, seed: int, variation: bool):
        """Best candidate over both engines: close to the prompt and typical of the draws."""
        pool = []
        for kind in KINDS:
            anchors = None
            if variation:
                near = self.nearest(c, types, kind, k=max(1, self.n // 2))
                anchors = [self.bank_vector(i) for i, _ in near for _ in range(2)][:self.n]
            x = self.draw(kind, c, types, seed, anchors)
            audio = np.stack([render(kind, v) for v in x])
            peak = np.abs(audio).reshape(len(x), -1).max(axis=1)
            z = self.hear(audio)
            match = self.match(z, self.type_logp(z), c, types)
            typical = z @ normalize(z.mean(axis=0))
            for k in range(len(x)):
                if peak[k] < 1e-3 or not np.isfinite(peak[k]):
                    continue  # silent or broken
                score = match[k] + TYPICAL_WEIGHT * typical[k]
                pool.append({"kind": kind, "vector": x[k], "match": float(match[k]), "typical": float(typical[k]),
                             "score": float(score), "audio": audio[k]})
        return sorted(pool, key=lambda r: -r["score"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=BASE_DIR / "data" / "v3" / "generations_v31")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--strength", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    gen = SynthGenerator(best_device(), args.candidates, args.guidance, args.strength)
    conds, types = gen.text(prompts)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (prompt, c, t) in enumerate(zip(prompts, conds, types)):
        row = {"prompt": prompt, "types": {TYPES[k]: round(float(t[k]), 2) for k in np.argsort(-t)[:3]}}
        for system, variation in (("generation", False), ("variation", True)):
            best = gen.answer(c, t, args.seed + i, variation)[0]
            row[system] = {"kind": best["kind"], "vector": best["vector"].tolist(), "match": best["match"],
                           "typical": best["typical"]}
        j, score = gen.nearest(c, t)[0]
        row["selection"] = {"id": gen.entries[j]["id"], "name": gen.entries[j]["name"], "kind": str(gen.bank_kind[j]),
                            "match": score}
        rows.append(row)
        print(f"{prompt[:34]:34s} {list(row['types'])[0]:7s} gen {row['generation']['kind']:6s} "
              f"{row['generation']['match']:.2f} | var {row['variation']['kind']:6s} {row['variation']['match']:.2f} "
              f"| sel {row['selection']['name'][:14]:14s} {row['selection']['kind']:6s} {score:.2f}", flush=True)
    (args.out / "generations.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
