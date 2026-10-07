"""Text -> sound "prior": maps a CLAP text embedding to the CLAP audio embedding it should sound like.

CLAP puts texts and sounds in one space but with a gap between the two clouds; the generators are
conditioned on audio embeddings, so a text embedding is first carried across. Trained on human pairs
only: FSD50K recordings with their titles and labels, and real presets with their names and categories.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Prior(nn.Module):
    def __init__(self, dim: int = 512, width: int = 1024, depth: int = 3):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU(), nn.Linear(width, dim)]
        self.blocks = nn.ModuleList(nn.Sequential(*layers[i:i + 4]) for i in range(0, len(layers), 4))

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        x = F.normalize(text, dim=-1)
        for block in self.blocks:
            x = x + block(x)
        return F.normalize(x, dim=-1)


def prior_loss(pred: torch.Tensor, audio: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """Cosine to the paired sound + in-batch contrastive term (the right sound among the batch)."""
    audio = F.normalize(audio, dim=-1)
    cos = 1.0 - (pred * audio).sum(-1).mean()
    logits = pred @ audio.T / temperature
    target = torch.arange(pred.shape[0], device=pred.device)
    return cos + F.cross_entropy(logits, target)
