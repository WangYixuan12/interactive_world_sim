import math

import torch
from einops import parse_shape, rearrange
from torch import nn


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    out = a[t]
    dtype = out.dtype
    while len(out.shape) < len(x_shape):
        out = out[..., None]
    return out + torch.zeros(x_shape, device=t.device, dtype=dtype)


def linear_beta_schedule(timesteps):
    """linear schedule, proposed in original ddpm paper"""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """cosine schedule

    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """sigmoid schedule

    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class EinopsWrapper(nn.Module):
    """Einops wrapper for a module"""

    def __init__(self, from_shape: str, to_shape: str, module: nn.Module):
        super().__init__()
        self.module = module
        self.from_shape = from_shape
        self.to_shape = to_shape

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """Forward pass"""
        axes_lengths = parse_shape(x, pattern=self.from_shape)
        x = rearrange(x, f"{self.from_shape} -> {self.to_shape}")
        x = self.module(x, *args, **kwargs)
        x = rearrange(x, f"{self.to_shape} -> {self.from_shape}", **axes_lengths)
        return x


def get_einops_wrapped_module(module, from_shape: str, to_shape: str):
    class WrappedModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.wrapper = EinopsWrapper(from_shape, to_shape, module(*args, **kwargs))

        def forward(self, x: torch.Tensor, *args, **kwargs):
            return self.wrapper(x, *args, **kwargs)

    return WrappedModule
