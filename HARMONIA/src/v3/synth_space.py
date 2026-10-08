"""Synth space: a small text <-> sound space learned on top of CLAP from human labels.

CLAP knows everyday sounds but barely knows synth vocabulary (its text side finds the right type of a
preset - bass, pad, lead... - for 25 % of them). Two adapters, one on CLAP's audio embedding and one on the
multilingual text encoder, are trained so that a preset's sound sits next to its own name and category,
with a type head (src/v3/vocabulary.py) on the shared space.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.v3.vocabulary import TYPES

DIM = 256


class Adapter(nn.Module):
    def __init__(self, dim_in: int = 512, dim_out: int = DIM, width: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim_in), nn.Linear(dim_in, width), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(width, width), nn.GELU(), nn.Linear(width, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(F.normalize(x, dim=-1)), dim=-1)


class SynthSpace(nn.Module):
    def __init__(self, dim: int = DIM):
        super().__init__()
        self.audio = Adapter(dim_out=dim)
        self.text = Adapter(dim_out=dim)
        self.types = nn.Linear(dim, len(TYPES))
        self.log_scale = nn.Parameter(torch.tensor(2.996))  # 1 / 0.05

    def type_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.types(z)


def contrastive_loss(text_z: torch.Tensor, audio_z: torch.Tensor, text_ids: torch.Tensor, scale: torch.Tensor):
    """Symmetric InfoNCE where every pair sharing the same text is a positive (many presets are called "Pads")."""
    logits = scale * text_z @ audio_z.T
    same = (text_ids[:, None] == text_ids[None, :]).float()
    target = same / same.sum(1, keepdim=True)
    return 0.5 * (torch.sum(-target * F.log_softmax(logits, 1), 1).mean()
                  + torch.sum(-target.T * F.log_softmax(logits.T, 1), 1).mean())


def type_loss(logits: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
    """Multi-label loss on the samples that have a type (others carry no type information)."""
    has = types.sum(1) > 0
    if not torch.any(has):
        return logits.sum() * 0
    return F.binary_cross_entropy_with_logits(logits[has], types[has])
