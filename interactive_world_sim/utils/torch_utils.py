"""This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research
template [repo](https://github.com/buoyancy99/research-template).
By its MIT license, you must keep the above sentence in `README.md`
and the `LICENSE` file to credit the author.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.types import _size


def freeze_model(model: nn.Module) -> None:
    """Freeze the torch model"""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def bernoulli_tensor(
    size: _size,
    p: float,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
):
    """Generate a tensor of the given size,
    where each element is sampled from a Bernoulli distribution with probability `p`.
    """
    return torch.bernoulli(torch.full(size, p, device=device), generator=generator)
