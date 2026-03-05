"""Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py
"""

import math
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from torch import nn


def get_activation(act_fn):
    if act_fn in ["swish", "silu"]:
        return nn.SiLU()
    elif act_fn == "mish":
        return nn.Mish()
    elif act_fn == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, dtype=dtype)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(
                cond_proj_dim, in_channels, bias=False, dtype=dtype
            )
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, dtype=dtype)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = sample.to(self.linear_1.weight.dtype)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.dtype = dtype

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            dtype=self.dtype,
        )
        return t_emb


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
    dtype: torch.dtype = torch.float32,
):
    """This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D or 2-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] or [N x M x dim] Tensor of positional embeddings.
    """
    if len(timesteps.shape) not in [1, 2]:
        raise ValueError("Timesteps should be a 1D or 2D tensor")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=dtype, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[..., None].float() * emb

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[..., half_dim:], emb[..., :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class RandomEmbeddingDropout(nn.Module):
    """Randomly nullify the input embeddings with a given probability."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, emb: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Randomly nullify the input embeddings with a probability p during training. For inference, the embeddings are nullified only if mask is provided.
        Args:
            emb: input embeddings of shape (B, ...)
            mask: mask tensor of shape (B, ). Only allowed during inference. If provided, embeddings for masked batches will be zeroed.
        """
        if mask is not None:
            # assert not self.training, "embedding mask is only allowed during inference"
            assert mask.ndim == 1, "embedding mask should be of shape (B,)"

        # if self.training and self.p > 0:
        #     mask = torch.rand(emb.shape[:1], device=emb.device) < self.p
        if mask is not None:
            mask = rearrange(mask, "... -> ..." + " 1" * (emb.ndim - 1))
            emb = torch.where(mask, torch.zeros_like(emb), emb)
        return emb


class RandomDropoutCondEmbedding(TimestepEmbedding):
    """A layer for processing conditions into embeddings, randomly dropping embeddings of each frame during training.
    NOTE: If dropout_prob is 0, it will fall back to `TimestepEmbedding`. We use this trick to ensure the backward compatibility with our previous checkpoints.
    """

    def __init__(
        self,
        cond_dim: int,
        cond_emb_dim: int,
        dropout_prob: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        self.dropout_prob = dropout_prob
        if dropout_prob == 0:
            super().__init__(cond_dim, cond_emb_dim, dtype=dtype)
        else:
            nn.Module.__init__(self)
            self.dropout = RandomEmbeddingDropout(p=dropout_prob)
            self.embedding = TimestepEmbedding(cond_dim, cond_emb_dim, dtype=dtype)

    def forward(self, cond: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if self.dropout_prob == 0:
            return super().forward(cond)
        return self.dropout(self.embedding(cond), mask)


class StochasticUnknownTimesteps(Timesteps):
    def __init__(
        self,
        num_channels: int,
        p: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(num_channels, dtype=dtype)
        self.unknown_token = (
            nn.Parameter(torch.randn(1, num_channels)) if p > 0.0 else None
        )
        self.p = p

    def forward(self, timesteps: torch.Tensor, mask: Optional[torch.Tensor] = None):
        t_emb = super().forward(timesteps)
        # if p == 0.0 - return original embeddings both during training and inference
        if self.p == 0.0:
            return t_emb

        # training or mask is None - randomly replace embeddings with unknown token with probability p
        # (mask can only be None for logging training visualization when using latents)
        # or if p == 1.0 - always replace embeddings with unknown token even during inference)
        if self.training or self.p == 1.0 or mask is None:
            mask = torch.rand(t_emb.shape[:-1], device=t_emb.device) < self.p
            mask = mask[..., None].expand_as(t_emb)
            return torch.where(mask, self.unknown_token, t_emb)

        # # inference with p < 1.0 - replace embeddings with unknown token only for masked timesteps
        # if mask is None:
        #     assert False, "mask should be provided when 0.0 < p < 1.0"
        mask = mask[..., None].expand_as(t_emb)
        return torch.where(mask, self.unknown_token, t_emb)


class StochasticTimeEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_embed_dim: int,
        use_fourier: bool = False,
        p: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.use_fourier = use_fourier
        if self.use_fourier:
            assert p == 0.0, "Fourier embeddings do not support stochastic timesteps"
        self.timesteps = (
            FourierEmbedding(dim, bandwidth=1)
            if use_fourier
            else StochasticUnknownTimesteps(dim, p)
        )
        self.embedding = TimestepEmbedding(dim, time_embed_dim, dtype=dtype)

    def forward(self, timesteps: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return self.embedding(
            self.timesteps(timesteps)
            if self.use_fourier
            else self.timesteps(timesteps, mask)
        )


class FourierEmbedding(torch.nn.Module):
    """Adapted from EDM2 - https://github.com/NVlabs/edm2/blob/38d5a70fe338edc8b3aac4da8a0cefbc4a057fb8/training/networks_edm2.py#L73"""

    def __init__(self, num_channels, bandwidth=1, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))
        self.dtype = dtype

    def forward(self, x):
        y = x.to(self.dtype)
        y = y[..., None] * self.freqs.to(self.dtype)
        y = y + self.phases.to(self.dtype)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)
