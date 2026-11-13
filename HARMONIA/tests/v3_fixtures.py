"""Tiny, offline v3 model directory and stand-ins for the text student and CLAP, for the run-time tests.

The generators are untrained (random weights, small width): what they draw is noise, which is enough to
check everything around them - valid presets, locks, the search, the server routes - without the 500 MB of
trained weights or a download.
"""

import hashlib
import json

import numpy as np
import torch

from src.presets import dx7_render
from src.v3 import codecs
from src.v3.diffusion import Diffusion
from src.v3.inference import HarmoniaV3, Settings
from src.v3.prior import Prior

DIM = 512


def build_model_dir(path):
    torch.manual_seed(0)
    (path / "generator").mkdir(parents=True)
    torch.save(Prior().state_dict(), path / "generator" / "prior.pt")
    for engine, dim in (("analog", codecs.ANALOG_DIM), ("dx7", codecs.DX7_DIM)):
        torch.save(Diffusion.create(dim, width=32, depth=1).model.state_dict(), path / "generator" / f"{engine}.pt")
    (path / "manifest.json").write_text(json.dumps({"model_version": "harmonia_v3", "files": {}}), encoding="utf-8")
    return path


def text_encoder(prompt: str) -> np.ndarray:
    """A unit vector that depends on the prompt only."""
    seed = int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest()[:4], "big")
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class Listener:
    """Stand-in for CLAP's audio tower: a fixed random projection of the sound's spectrum and envelope."""

    def __init__(self):
        self.projection = np.random.default_rng(1).standard_normal((96, DIM)).astype(np.float32)

    def embed_audio(self, waves, batch_size: int = 16) -> np.ndarray:
        out = []
        for wave in waves:
            wave = np.asarray(wave, dtype=np.float32)
            spectrum = np.abs(np.fft.rfft(wave[:65536], 65536))[:32768].reshape(64, -1).mean(axis=1)
            envelope = np.abs(wave[:len(wave) // 32 * 32]).reshape(32, -1).mean(axis=1)
            features = np.log1p(100.0 * np.concatenate([spectrum, envelope * 64]))
            v = features @ self.projection
            out.append(v / max(float(np.linalg.norm(v)), 1e-8))
        return np.stack(out).astype(np.float32) if out else np.zeros((0, DIM), dtype=np.float32)


def build_engine(path, **settings) -> HarmoniaV3:
    defaults = dict(candidates=3, polish_rounds=1, refine_rounds=2, population=4)
    defaults.update(settings)
    return HarmoniaV3(build_model_dir(path), settings=Settings(**defaults), listener=Listener(),
                      text_encoder=text_encoder)


def dx7_available() -> bool:
    return dx7_render.LIBRARY.exists()
