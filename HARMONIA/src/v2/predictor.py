"""Text embedding (512, CLAP space) -> charter parameters (20, normalized)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.charter import CHARTER, CONTINUOUS_INDICES, BIPOLAR_INDICES, DISCRETE_INDICES, DISCRETE_STEPS

PARAM_COUNT = len(CHARTER)


class ParamPredictor(nn.Module):
    """MLP with charter-aware heads: sigmoid for continuous/bipolar values, softmax over steps for discrete ones."""

    def __init__(self, embed_dim: int = 512, hidden: int = 768, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.continuous = nn.Linear(hidden, len(CONTINUOUS_INDICES) + len(BIPOLAR_INDICES))
        self.discrete = nn.ModuleList([nn.Linear(hidden, len(DISCRETE_STEPS[i])) for i in DISCRETE_INDICES])
        self.register_buffer("cont_index", torch.tensor(list(CONTINUOUS_INDICES) + list(BIPOLAR_INDICES)), persistent=False)

    def forward(self, emb: torch.Tensor):
        """Returns (values (B, 20) in [0, 1], list of discrete logits)."""
        x = self.trunk(emb)
        values = torch.zeros(emb.shape[0], PARAM_COUNT, dtype=x.dtype, device=x.device)
        values[:, self.cont_index] = torch.sigmoid(self.continuous(x))
        logits = []
        for head, idx in zip(self.discrete, DISCRETE_INDICES):
            lg = head(x)
            logits.append(lg)
            steps = torch.linspace(0.0, 1.0, lg.shape[-1], device=x.device, dtype=x.dtype)
            values[:, idx] = steps[lg.argmax(dim=-1)] if not self.training else (F.softmax(lg, dim=-1) * steps).sum(-1)
        return values, logits


class ExportablePredictor(nn.Module):
    """Inference wrapper (ONNX): returns only the snapped (B, 20) parameter vector."""

    def __init__(self, predictor: ParamPredictor):
        super().__init__()
        self.predictor = predictor.eval()

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        values, _ = self.predictor(emb)
        return values
