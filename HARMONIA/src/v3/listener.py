"""CLAP's audio side only, for the v3 run time: how a rendered preset sounds, as a 512-d vector.

The training scripts use src/v2/clap_audio.ClapEmbedder (text and audio towers, from the Hugging Face hub).
The server only listens: this loads the audio tower from the model directory (clap_audio/, written by
scripts/v3/export_model.py) and falls back to the hub when the directory does not hold it. Same mel
front-end and the same embeddings as the embedder the generators were trained with.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from src.v2.clap_audio import TEACHER_ID, TEACHER_REVISION, ClapEmbedder, fit_to_window

AUDIO_DIR = "clap_audio"


class ClapListener(ClapEmbedder):
    def __init__(self, model_dir: Optional[Path] = None, device: Optional[torch.device] = None):
        from transformers import ClapAudioModelWithProjection, ClapFeatureExtractor

        local = Path(model_dir) / AUDIO_DIR if model_dir is not None else None
        if local is not None and (local / "config.json").exists():
            source, kwargs = str(local), {}
        else:
            source, kwargs = TEACHER_ID, {"revision": TEACHER_REVISION}
        self.device = device or torch.device("cpu")
        self.dtype = torch.float32
        # local files, or the hub at a pinned revision
        fe = ClapFeatureExtractor.from_pretrained(source, **kwargs)  # nosec B615
        self.model = ClapAudioModelWithProjection.from_pretrained(source, **kwargs).float().to(self.device).eval()  # nosec B615
        self.n_fft = int(fe.fft_window_size)
        self.hop = int(fe.hop_length)
        self.mel = torch.tensor(fe.mel_filters_slaney, dtype=torch.float32, device=self.device)
        self.window = torch.hann_window(self.n_fft, periodic=True, device=self.device)

    @torch.no_grad()
    def embed_audio(self, waves: Sequence[np.ndarray], batch_size: int = 16) -> np.ndarray:
        """48 kHz mono clips -> L2-normalized (N, 512) float32 embeddings."""
        out: List[np.ndarray] = []
        for i in range(0, len(waves), batch_size):
            chunk = np.stack([fit_to_window(w) for w in waves[i:i + batch_size]])
            x = torch.from_numpy(chunk).to(self.device)
            features = self.log_mel(x).unsqueeze(1)
            is_longer = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=self.device)
            emb = self.model(input_features=features, is_longer=is_longer).audio_embeds
            out.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), dtype=np.float32)

    def embed_text(self, texts, batch_size: int = 256):  # pragma: no cover - the run time has no text tower
        raise NotImplementedError("the listener only hears: prompts go through the text student")
