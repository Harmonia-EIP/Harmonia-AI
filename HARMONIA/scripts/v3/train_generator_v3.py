#!/usr/bin/env python3
"""Train the v3 preset generators and the text -> sound prior.

- DX7 generator and analog generator: diffusion models over preset vectors (src/v3/codecs.py), conditioned
  on the CLAP audio embedding of each real preset of the bank (scripts/v3/build_bank_v3.py).
- Prior (src/v3/prior.py): CLAP text embedding -> CLAP audio embedding, trained on human pairs only:
  FSD50K recordings with their title / class label, and bank presets with their own names and categories.

    python scripts/v3/train_generator_v3.py      # -> data/v3/generator/{dx7,analog,prior}.pt + meta.json
    python scripts/v3/train_generator_v3.py --cond synth --out data/v3/generator_synth   # v3.1
"""

from __future__ import annotations

import argparse
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
from src.presets import dx7_render  # noqa: E402
from src.presets.labels import preset_text  # noqa: E402
from src.v2.clap_audio import best_device  # noqa: E402
from src.v3 import codecs  # noqa: E402
from src.v3.diffusion import Diffusion  # noqa: E402
from src.v3.prior import Prior, prior_loss  # noqa: E402

BANK = BASE_DIR / "data" / "v3" / "bank"
OUT = BASE_DIR / "data" / "v3" / "generator"
SYNTH_SPACE = BASE_DIR / "data" / "v3" / "synth_space"
FSD50K_DEV = BASE_DIR.parents[1] / "ai-v2" / "HARMONIA" / "data" / "v2" / "fsd50k_dev.npz"
COND_NOISE = 0.05  # the prior's outputs are never exactly a real preset's embedding
LABELLED_WEIGHT = 2.0  # named presets (see the synth branch of main)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def load_bank():
    z = np.load(BANK / "bank.npz")
    entries = [json.loads(line) for line in open(BANK / "entries.jsonl", encoding="utf-8")]
    emb = normalize(z["emb"].astype(np.float32))
    return z, entries, emb


class SoundConditions:
    """CLAP embedding of each preset's sound, one note at a time (v3.0)."""

    def __init__(self, cond: np.ndarray, device):
        self.ct = torch.from_numpy(cond).float().to(device)  # (n, notes, 512)
        self.dim = self.ct.shape[-1]

    def __call__(self, idx: np.ndarray, rng) -> torch.Tensor:
        note = torch.from_numpy(rng.integers(0, self.ct.shape[1], len(idx))).to(self.ct.device)
        c = self.ct[torch.from_numpy(idx).to(self.ct.device), note]
        return torch.nn.functional.normalize(c + COND_NOISE * torch.randn_like(c), dim=-1)


class SynthConditions:
    """Synth-space conditions (scripts/v3/train_synth_space.py): half the time the preset's sound, half the
    time one of its own texts (name, category, comment, their French versions), so the generator answers a
    prompt directly instead of going through a text -> sound prior. The condition ends with the type of
    the sound (bass, pad, ...): the preset's labelled type when it has one, else the type the synth space
    recognises; at generation time, the type read in the prompt."""

    def __init__(self, sc, rows: np.ndarray, device, text_share: float = 0.5):
        self.audio = torch.from_numpy(sc["audio_z"][rows].astype(np.float32)).to(device)  # (n, notes, 256)
        self.audio_types = torch.from_numpy(sc["audio_types"][rows].astype(np.float32)).to(device)
        self.text = torch.from_numpy(sc["text_z"].astype(np.float32)).to(device)
        self.text_types = torch.from_numpy(sc["text_types"].astype(np.float32)).to(device)
        labels = sc["labels"][rows].astype(np.float32)
        self.labels = torch.from_numpy(labels).to(device)
        self.labelled = torch.from_numpy(labels.sum(1) > 0).to(device)
        local = {int(r): k for k, r in enumerate(rows)}
        owners = np.array([local.get(int(o), -1) for o in sc["text_owner"]])
        order = np.flatnonzero(owners >= 0)
        order = order[np.argsort(owners[order], kind="stable")]
        self.text_rows = order
        counts = np.bincount(owners[order], minlength=len(rows))
        self.first = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.count = counts
        self.share = text_share
        self.dim = self.audio.shape[-1] + self.labels.shape[-1]

    def __call__(self, idx: np.ndarray, rng) -> torch.Tensor:
        device = self.audio.device
        rows = torch.from_numpy(idx).to(device)
        note = torch.from_numpy(rng.integers(0, self.audio.shape[1], len(idx))).to(device)
        z = self.audio[rows, note]
        types = self.audio_types[rows, note]
        use_text = (rng.random(len(idx)) < self.share) & (self.count[idx] > 0)
        if use_text.any():
            pick = self.first[idx[use_text]] + (rng.random(use_text.sum()) * self.count[idx[use_text]]).astype(int)
            where = torch.from_numpy(np.flatnonzero(use_text)).to(device)
            text = torch.from_numpy(self.text_rows[pick]).to(device)
            z[where] = self.text[text]
            types[where] = self.text_types[text]
        types = torch.where(self.labelled[rows][:, None], self.labels[rows], types)
        z = torch.nn.functional.normalize(z + COND_NOISE * torch.randn_like(z), dim=-1)
        return torch.cat([z, types], dim=-1)


def train_diffusion(name: str, x: np.ndarray, conditions, steps: int, batch: int, device, seed: int,
                    weights: np.ndarray | None = None):
    torch.manual_seed(seed)
    diffusion = Diffusion.create(x.shape[1], cond_dim=conditions.dim)
    model = diffusion.model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    xt = torch.from_numpy(x).float().to(device)
    rng = np.random.default_rng(seed)
    start, running = time.time(), 0.0
    for step in range(1, steps + 1):
        rows = rng.integers(0, x.shape[0], batch) if weights is None else rng.choice(x.shape[0], batch, p=weights)
        idx = torch.from_numpy(rows).to(device)
        c = conditions(rows, rng)
        loss = diffusion.loss(xt[idx], c)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 1000 == 0 or step == steps:
            print(f"{name} step {step}/{steps} loss {running:.4f} ({time.time() - start:.0f}s)", flush=True)
    model.cpu()
    return diffusion


def prior_pairs(entries, emb, embed_text):
    """(text, audio embedding) pairs: FSD50K clips with their own words, presets with their own names."""
    texts, audio = [], []
    fsd = np.load(FSD50K_DEV)
    fsd_emb = normalize(fsd["emb"].astype(np.float32))
    for title, labels, e in zip(fsd["title"], fsd["labels"], fsd_emb):
        title = clean_title(str(title))
        if len(title.split()) >= 2:
            texts.append(title)
            audio.append(e)
        texts.append(label_phrase(str(labels).split(",")[0]))
        audio.append(e)
    for entry, e in zip(entries, emb.mean(axis=1)):
        text = preset_text(entry)
        if text:
            texts.append(text)
            audio.append(e)
    unique = sorted(set(texts))
    table = dict(zip(unique, embed_text(unique)))
    return np.stack([table[t] for t in texts]).astype(np.float32), normalize(np.stack(audio)).astype(np.float32)


def train_prior(text_emb, audio_emb, epochs: int, batch: int, device, seed: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(text_emb))
    val, train = order[:2000], order[2000:]
    model = Prior().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    tt = torch.from_numpy(text_emb).to(device)
    at = torch.from_numpy(audio_emb).to(device)
    report = {}
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(train)
        for i in range(0, len(perm) - batch + 1, batch):
            idx = torch.from_numpy(perm[i:i + batch]).to(device)
            loss = prior_loss(model(tt[idx]), at[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vi = torch.from_numpy(val).to(device)
            pred = model(tt[vi])
            cos = (pred * at[vi]).sum(-1).mean().item()
            base = (torch.nn.functional.normalize(tt[vi], dim=-1) * at[vi]).sum(-1).mean().item()
            ranks = (pred @ at[vi].T).argsort(dim=1, descending=True)
            r10 = (ranks[:, :10] == torch.arange(len(val), device=device)[:, None]).any(1).float().mean().item()
        report = {"epoch": epoch + 1, "val_cos": round(cos, 4), "text_audio_cos_before": round(base, 4),
                  "val_recall@10_of_2000": round(r10, 4)}
        print("prior", json.dumps(report), flush=True)
    model.cpu()
    return model, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx7-steps", type=int, default=40000)
    parser.add_argument("--analog-steps", type=int, default=15000)
    parser.add_argument("--prior-epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cond", choices=("clap", "synth"), default="clap",
                        help="clap: v3.0 (CLAP sound + prior); synth: synth space, prompts used directly")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    from src.v2.clap_audio import TEACHER_ID, TEACHER_REVISION, ClapEmbedder

    device = best_device()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    z, entries, emb = load_bank()
    kind = z["kind"]
    dx7_idx = np.flatnonzero(kind == "dx7")
    analog_idx = np.flatnonzero(kind == "analog")
    print(f"bank: {len(dx7_idx)} DX7 voices, {len(analog_idx)} analog presets", flush=True)

    x_dx7 = np.stack([codecs.dx7_encode(dx7_render.from_patch(bytes(z["dx7"][i]))) for i in dx7_idx])
    x_analog = np.stack([codecs.analog_encode(z["analog"][i].astype(np.float64)) for i in analog_idx]).astype(np.float32)

    if args.cond == "synth":
        sc = np.load(SYNTH_SPACE / "bank_cond.npz")
        meta = {"cond": "synth", "cond_dim": int(sc["audio_z"].shape[-1] + sc["labels"].shape[-1]),
                "dx7_presets": int(len(dx7_idx)),
                "analog_presets": int(len(analog_idx)), "args": {k: str(v) for k, v in vars(args).items()}}
        for name, rows, x, steps, batch in (("dx7", dx7_idx, x_dx7, args.dx7_steps, 512),
                                            ("analog", analog_idx, x_analog, args.analog_steps, 256)):
            cond = SynthConditions(sc, rows, device)
            # voices whose names say what they are (E.PIANO, BRASS...) come up twice as often: the cartridge
            # collection also holds many unnamed experiments
            labelled = sc["labels"][rows].sum(1) > 0
            weights = np.where(labelled, LABELLED_WEIGHT, 1.0)
            model = train_diffusion(name, x, cond, steps, batch, device, args.seed, weights / weights.sum())
            torch.save(model.model.state_dict(), out / f"{name}.pt")
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2))
        return 0

    embedder = ClapEmbedder(half=True)
    text_emb, audio_emb = prior_pairs(entries, emb, lambda t: normalize(embedder.embed_text(t)))
    del embedder
    print(f"prior pairs: {len(text_emb)}", flush=True)
    prior, prior_report = train_prior(text_emb, audio_emb, args.prior_epochs, 512, device, args.seed)
    torch.save(prior.state_dict(), out / "prior.pt")

    dx7 = train_diffusion("dx7", x_dx7, SoundConditions(emb[dx7_idx], device), args.dx7_steps, 512, device, args.seed)
    torch.save(dx7.model.state_dict(), out / "dx7.pt")
    analog = train_diffusion("analog", x_analog, SoundConditions(emb[analog_idx], device), args.analog_steps, 256,
                             device, args.seed)
    torch.save(analog.model.state_dict(), out / "analog.pt")

    meta = {"teacher": {"id": TEACHER_ID, "revision": TEACHER_REVISION}, "dx7_dim": codecs.DX7_DIM,
            "analog_dim": codecs.ANALOG_DIM, "dx7_presets": int(len(dx7_idx)), "analog_presets": int(len(analog_idx)),
            "prior_pairs": int(len(text_emb)), "prior": prior_report, "cond_noise": COND_NOISE, "args": vars(args)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
