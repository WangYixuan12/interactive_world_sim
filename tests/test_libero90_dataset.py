from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from interactive_world_sim.datasets.latent_dynamics.libero90_dataset import (
    Libero90Dataset,
)


def _write_task(path: Path, task_value: int) -> None:
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        for demo_idx in range(3):
            demo = data.create_group(f"demo_{demo_idx}")
            obs = demo.create_group("obs")
            value = task_value + demo_idx
            obs.create_dataset(
                "agentview_rgb",
                data=np.full((2, 128, 128, 3), value, dtype=np.uint8),
            )
            obs.create_dataset(
                "eye_in_hand_rgb",
                data=np.full((2, 128, 128, 3), value + 10, dtype=np.uint8),
            )
            demo.create_dataset(
                "actions", data=np.full((2, 7), value, dtype=np.float32)
            )


def test_libero90_dataset_splits_tasks_and_returns_two_views(tmp_path: Path) -> None:
    _write_task(tmp_path / "task_a.hdf5", 1)
    _write_task(tmp_path / "task_b.hdf5", 20)
    cfg = OmegaConf.create(
        {
            "dataset_dir": str(tmp_path),
            "expected_tasks": 2,
            "val_demos_per_task": 1,
            "resolution": 128,
            "action_dim": 7,
            "obs_keys": ["agentview_rgb", "eye_in_hand_rgb"],
        }
    )

    train = Libero90Dataset(cfg)
    validation = train.get_validation_dataset()
    sample = train[0]
    batched_views = [sample["obs"][key].unsqueeze(0) for key in cfg.obs_keys]

    assert len(train) == 8
    assert len(validation) == 4
    assert sample["obs"]["agentview_rgb"].shape == (1, 3, 128, 128)
    assert sample["obs"]["eye_in_hand_rgb"].shape == (1, 3, 128, 128)
    assert sample["action"].shape == (1, 7)
    assert torch.cat(batched_views, dim=2).shape == (1, 1, 6, 128, 128)
    assert torch.isclose(
        sample["obs"]["agentview_rgb"][0, 0, 0, 0], torch.tensor(1 / 255)
    )
    assert validation[0]["action"][0, 0].item() == 3
