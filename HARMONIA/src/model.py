from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.charter import (
    BIPOLAR_INDICES,
    CHARTER,
    CONTINUOUS_INDICES,
    DISCRETE_INDICES,
    DISCRETE_STEPS,
)


MODEL_ID = os.environ.get("HARMONIA_MODEL_ID", "prajjwal1/bert-tiny")
MODEL_REVISION = os.environ.get("HARMONIA_MODEL_REVISION", "main")

CHARTER_PARAM_COUNT = len(CHARTER)


def _masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class _CharterHead(nn.Module):

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.continuous_head = nn.Linear(256, len(CONTINUOUS_INDICES))
        self.bipolar_head = nn.Linear(256, len(BIPOLAR_INDICES)) if BIPOLAR_INDICES else None
        self.discrete_heads = nn.ModuleList(
            [nn.Linear(256, len(DISCRETE_STEPS[idx])) for idx in DISCRETE_INDICES]
        )
        self._discrete_step_centres = {
            idx: torch.linspace(0.0, 1.0, steps=len(DISCRETE_STEPS[idx]))
            for idx in DISCRETE_INDICES
        }

    def _step_centres(self, idx: int, device: torch.device) -> torch.Tensor:
        cached = self._discrete_step_centres[idx]
        if cached.device != device:
            cached = cached.to(device)
            self._discrete_step_centres[idx] = cached
        return cached

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        x = self.shared(self.norm(pooled))
        batch = x.size(0)
        device = x.device
        out = torch.empty(batch, CHARTER_PARAM_COUNT, device=device, dtype=x.dtype)

        cont_logits = self.continuous_head(x)
        cont_values = torch.sigmoid(cont_logits)
        for slot, idx in enumerate(CONTINUOUS_INDICES):
            out[:, idx] = cont_values[:, slot]

        if self.bipolar_head is not None:
            bip_values = torch.sigmoid(self.bipolar_head(x))
            for slot, idx in enumerate(BIPOLAR_INDICES):
                out[:, idx] = bip_values[:, slot]

        for slot, idx in enumerate(DISCRETE_INDICES):
            logits = self.discrete_heads[slot](x)
            probs = F.softmax(logits, dim=-1)
            centres = self._step_centres(idx, device)
            out[:, idx] = (probs * centres).sum(dim=-1)

        return out


class _LegacyHead(nn.Module):
    def __init__(self, hidden_size: int, num_plugin_parameters: int, output_activation: str):
        super().__init__()
        layers = [
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_plugin_parameters),
        ]
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class TextToParams(nn.Module):
    def __init__(self, num_plugin_parameters: int = CHARTER_PARAM_COUNT, output_activation: str = "sigmoid"):
        super().__init__()
        if int(num_plugin_parameters) <= 0:
            raise ValueError("num_plugin_parameters must be > 0")

        self.bert = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION)  # nosec B615
        hidden_size = int(getattr(self.bert.config, "hidden_size", 128))
        self.num_plugin_parameters = int(num_plugin_parameters)

        if self.num_plugin_parameters == CHARTER_PARAM_COUNT:
            self.head: nn.Module = _CharterHead(hidden_size)
            self._charter_mode = True
        else:
            self.head = _LegacyHead(hidden_size, self.num_plugin_parameters, output_activation)
            self._charter_mode = False

    @property
    def charter_mode(self) -> bool:
        return self._charter_mode

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if self._charter_mode:
            pooled = _masked_mean_pool(outputs.last_hidden_state, attention_mask)
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        return self.head(pooled)
