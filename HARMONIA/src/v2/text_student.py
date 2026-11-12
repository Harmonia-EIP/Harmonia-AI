"""Multilingual student text encoder distilled into the CLAP text embedding space."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

STUDENT_BASE_ID = os.environ.get("HARMONIA_STUDENT_BASE", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
STUDENT_BASE_REVISION = os.environ.get("HARMONIA_STUDENT_BASE_REVISION", "e8f8c211226b894fcb81acc59f3b34ba3efd5f42")
EMBED_DIM = 512
MAX_TOKENS = 64


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class TextStudent(nn.Module):
    def __init__(self, base_id: str = STUDENT_BASE_ID, revision: str = STUDENT_BASE_REVISION, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_id, revision=revision)  # nosec B615
        self.proj = nn.Linear(self.encoder.config.hidden_size, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return F.normalize(self.proj(masked_mean(hidden, attention_mask)), dim=-1)


def load_student_tokenizer(base_id: str = STUDENT_BASE_ID, revision: str = STUDENT_BASE_REVISION):
    return AutoTokenizer.from_pretrained(base_id, revision=revision)  # nosec B615


def load_trained_student(path, device: torch.device):
    """Load a student saved by train_text_encoder.py: returns (model, tokenizer)."""
    path = Path(path)
    model = TextStudent(base_id=str(path / "encoder"), revision="main")
    model.proj.load_state_dict(torch.load(path / "proj.pt", map_location="cpu", weights_only=True))
    tokenizer = AutoTokenizer.from_pretrained(str(path / "encoder"))  # nosec B615
    return model.to(device).eval(), tokenizer


@torch.no_grad()
def encode_texts(model: TextStudent, tokenizer, sentences, device: torch.device, batch: int = 256):
    out = []
    for i in range(0, len(sentences), batch):
        tok = tokenizer(list(sentences[i : i + batch]), padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
        out.append(model(tok["input_ids"].to(device), tok["attention_mask"].to(device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, EMBED_DIM), dtype=np.float32)
