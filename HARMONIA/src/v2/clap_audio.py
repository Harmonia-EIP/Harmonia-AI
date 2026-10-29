"""Batched CLAP embeddings (training-time only, needs torch + transformers).

The audio front-end reproduces ClapFeatureExtractor ("rand_trunc" + "repeatpad", Slaney mel
filters, power dB) with torch.stft so whole batches run on the GPU (MPS/CUDA) instead of numpy.
Clips longer than 10 s are cropped deterministically around their loudest 10 s window.
"""

from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

TEACHER_ID = os.environ.get("HARMONIA_CLAP_TEACHER", "laion/larger_clap_general")
TEACHER_REVISION = os.environ.get("HARMONIA_CLAP_TEACHER_REVISION", "ada0c23a36c4e8582805bb38fec3905903f18b41")
CLAP_SR = 48000
MAX_SAMPLES = 10 * CLAP_SR


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_to_window(wave: np.ndarray, max_samples: int = MAX_SAMPLES) -> np.ndarray:
    """Repeat-pad short clips (as CLAP was trained) and crop long ones to their loudest window."""
    wave = np.asarray(wave, dtype=np.float32)
    n = wave.shape[0]
    if n == 0:
        return np.zeros(max_samples, dtype=np.float32)
    if n > max_samples:
        energy = np.cumsum(np.concatenate([[0.0], wave.astype(np.float64) ** 2]))
        hop = CLAP_SR // 10
        starts = np.arange(0, n - max_samples + 1, hop)
        window_energy = energy[starts + max_samples] - energy[starts]
        start = int(starts[int(np.argmax(window_energy))])
        return wave[start : start + max_samples]
    repeats = max_samples // n
    tiled = np.tile(wave, repeats)
    return np.pad(tiled, (0, max_samples - tiled.shape[0]))


class ClapEmbedder:
    def __init__(
        self,
        model_id: str = TEACHER_ID,
        revision: str = TEACHER_REVISION,
        device: torch.device | None = None,
        half: bool = False,
    ):
        self.device = device or best_device()
        self.processor = ClapProcessor.from_pretrained(model_id, revision=revision)  # nosec B615
        self.model = ClapModel.from_pretrained(model_id, revision=revision).to(self.device).eval()  # nosec B615
        # fp16 is ~20% faster on MPS and gives the same embeddings (cosine > 0.9999 vs fp32).
        self.dtype = torch.float16 if half and self.device.type != "cpu" else torch.float32
        self.model.to(self.dtype)
        fe = self.processor.feature_extractor
        self.n_fft = int(fe.fft_window_size)
        self.hop = int(fe.hop_length)
        self.mel = torch.tensor(fe.mel_filters_slaney, dtype=torch.float32, device=self.device)
        self.window = torch.hann_window(self.n_fft, periodic=True, device=self.device)

    def log_mel(self, waves: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            waves,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spec.real.pow(2) + spec.imag.pow(2)            # (B, 513, frames)
        mel = torch.einsum("bft,fm->btm", power, self.mel)      # (B, frames, 64)
        return 10.0 * torch.log10(mel.clamp(min=1e-10))

    @torch.no_grad()
    def embed_audio(self, waves: Sequence[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """48 kHz mono clips -> L2-normalized (N, 512) float32 embeddings."""
        out: List[np.ndarray] = []
        for i in range(0, len(waves), batch_size):
            chunk = np.stack([fit_to_window(w) for w in waves[i : i + batch_size]])
            x = torch.from_numpy(chunk).to(self.device)
            features = self.log_mel(x).unsqueeze(1).to(self.dtype)
            is_longer = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=self.device)
            emb = self.model.get_audio_features(input_features=features, is_longer=is_longer)
            emb = emb.pooler_output if hasattr(emb, "pooler_output") else emb
            out.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), dtype=np.float32)

    @torch.no_grad()
    def embed_text(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        out: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            tok = self.processor.tokenizer(list(texts[i : i + batch_size]), padding=True, truncation=True, return_tensors="pt")
            tok = {k: v.to(self.device) for k, v in tok.items()}
            emb = self.model.get_text_features(**tok)
            emb = emb.pooler_output if hasattr(emb, "pooler_output") else emb
            out.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), dtype=np.float32)
