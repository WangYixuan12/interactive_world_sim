from types import SimpleNamespace

import torch
from torch.nn.attention import SDPBackend

from interactive_world_sim.algorithms.models.attention import Attention


def test_a100_attention_falls_back_to_math(monkeypatch) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(major=8, minor=0),
    )

    attention = Attention(query_dim=8)

    assert attention.cuda_backends == [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.MATH,
    ]
