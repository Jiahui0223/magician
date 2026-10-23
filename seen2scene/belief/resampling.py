"""Resampling of particle weights (deterministic with a torch.Generator; computed in float64 on the CPU).

All schemes return K ancestor indices in ascending order, so particles keep their relative order.
- systematic: one uniform offset, K evenly spaced points on the CDF (lowest variance, the default);
- residual: floor(K w) deterministic copies, the remainder multinomial;
- multinomial: K independent draws.
"""
from typing import Optional

import torch


def _cdf(weights: torch.Tensor) -> torch.Tensor:
    w = weights.detach().double().cpu()
    w = w / w.sum()
    c = torch.cumsum(w, 0)
    c[-1] = 1.0
    return c


def systematic(weights: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    K = weights.numel()
    u = (torch.rand(1, generator=generator, dtype=torch.float64) + torch.arange(K, dtype=torch.float64)) / K
    return torch.searchsorted(_cdf(weights), u, right=True).clamp(max=K - 1)


def multinomial(weights: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    K = weights.numel()
    u = torch.rand(K, generator=generator, dtype=torch.float64)
    return torch.searchsorted(_cdf(weights), u, right=True).clamp(max=K - 1).sort().values


def residual(weights: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    K = weights.numel()
    w = weights.detach().double().cpu()
    w = w / w.sum()
    n = torch.floor(K * w).long()
    idx = torch.repeat_interleave(torch.arange(K), n)
    rest = K - int(n.sum())
    if rest > 0:
        r = K * w - n.double()
        idx = torch.cat([idx, multinomial(r / r.sum(), generator)[:rest]])
    return idx.sort().values


SCHEMES = {"systematic": systematic, "multinomial": multinomial, "residual": residual}
