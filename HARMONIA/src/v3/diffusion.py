"""Small conditional diffusion model over preset vectors (src/v3/codecs.py).

A denoising MLP predicts the noise added to a preset vector, given the noise level and a condition
(a CLAP embedding of how the preset sounds). Sampling (DDIM) draws presets that look like real presets
for that condition instead of averaging them, which is what made v2's direct predictor sound dull.
Classifier-free guidance: the condition is dropped for 10 % of training examples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

STEPS = 1000


def cosine_alphas_cumprod(steps: int = STEPS) -> torch.Tensor:
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    return (f[1:] / f[0]).clamp(1e-5, 0.9999).float()


class Block(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Denoiser(nn.Module):
    def __init__(self, dim: int, cond_dim: int = 512, width: int = 512, depth: int = 4):
        super().__init__()
        self.dim, self.cond_dim = dim, cond_dim
        self.inp = nn.Linear(dim, width)
        self.time = nn.Sequential(nn.Linear(128, width), nn.SiLU(), nn.Linear(width, width))
        self.cond = nn.Sequential(nn.Linear(cond_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.null = nn.Parameter(torch.zeros(cond_dim))
        self.blocks = nn.ModuleList(Block(width) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))

    @staticmethod
    def time_embedding(t: torch.Tensor) -> torch.Tensor:
        freqs = torch.exp(-math.log(10000.0) * torch.arange(64, device=t.device) / 64)
        angles = t.float()[:, None] * freqs[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.inp(x) + self.time(self.time_embedding(t)) + self.cond(cond)
        for block in self.blocks:
            h = block(h)
        return self.out(h)


@dataclass
class Diffusion:
    model: Denoiser
    alphas_cumprod: torch.Tensor

    @classmethod
    def create(cls, dim: int, cond_dim: int = 512, width: int = 512, depth: int = 4) -> "Diffusion":
        return cls(Denoiser(dim, cond_dim, width, depth), cosine_alphas_cumprod())

    def loss(self, x0: torch.Tensor, cond: torch.Tensor, drop: float = 0.1) -> torch.Tensor:
        b = x0.shape[0]
        device = x0.device
        t = torch.randint(0, STEPS, (b,), device=device)
        a = self.alphas_cumprod.to(device)[t][:, None]
        noise = torch.randn_like(x0)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        keep = (torch.rand(b, device=device) > drop)[:, None]
        cond = torch.where(keep, cond, self.model.null[None, :].expand_as(cond))
        return ((self.model(xt, t, cond) - noise) ** 2).mean()

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: int = 50, guidance: float = 2.0,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """DDIM (eta 0) with classifier-free guidance; cond is (n, cond_dim)."""
        device = cond.device
        n = cond.shape[0]
        ac = self.alphas_cumprod.to(device)
        x = torch.randn((n, self.model.dim), device=device, generator=generator)
        times = torch.linspace(STEPS - 1, 0, steps, device=device).long()
        null = self.model.null[None, :].expand_as(cond)
        for k, t in enumerate(times):
            tt = t.repeat(n)
            eps_c = self.model(x, tt, cond)
            eps_u = self.model(x, tt, null)
            eps = eps_u + guidance * (eps_c - eps_u)
            a = ac[t]
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1.5, 1.5)
            a_next = ac[times[k + 1]] if k + 1 < steps else torch.tensor(1.0, device=device)
            x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
        return x
