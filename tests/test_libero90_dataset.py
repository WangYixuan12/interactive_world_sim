from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

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


def _write_window_task(path: Path, lengths: tuple[int, ...]) -> None:
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        for demo_idx, length in enumerate(lengths):
            base = demo_idx * 100
            values = np.arange(base, base + length, dtype=np.uint8)
            images = np.broadcast_to(
                values[:, None, None, None], (length, 128, 128, 3)
            )
            demo = data.create_group(f"demo_{demo_idx}")
            obs = demo.create_group("obs")
            obs.create_dataset("agentview_rgb", data=images)
            obs.create_dataset("eye_in_hand_rgb", data=images + 10)
            demo.create_dataset(
                "actions",
                data=np.repeat(
                    np.arange(base, base + length, dtype=np.float32)[:, None],
                    7,
                    axis=1,
                ),
            )


def _window_cfg(path: Path, horizon: int, val_horizon: int):
    return OmegaConf.create(
        {
            "dataset_dir": str(path),
            "expected_tasks": 1,
            "val_demos_per_task": 1,
            "horizon": horizon,
            "val_horizon": val_horizon,
            "resolution": 128,
            "action_dim": 7,
            "obs_keys": ["agentview_rgb", "eye_in_hand_rgb"],
        }
    )


def test_libero90_dataset_splits_tasks_and_returns_two_views(tmp_path: Path) -> None:
    _write_task(tmp_path / "task_a.hdf5", 1)
    _write_task(tmp_path / "task_b.hdf5", 20)
    cfg = OmegaConf.create(
        {
            "dataset_dir": str(tmp_path),
            "expected_tasks": 2,
            "val_demos_per_task": 1,
            "horizon": 1,
            "val_horizon": 1,
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
    assert len(validation) == 2
    assert sample["obs"]["agentview_rgb"].shape == (1, 3, 128, 128)
    assert sample["obs"]["eye_in_hand_rgb"].shape == (1, 3, 128, 128)
    assert sample["action"].shape == (1, 7)
    assert torch.cat(batched_views, dim=2).shape == (1, 1, 6, 128, 128)
    assert torch.isclose(
        sample["obs"]["agentview_rgb"][0, 0, 0, 0], torch.tensor(1 / 255)
    )
    assert validation[0]["action"][0, 0].item() == 3


def test_libero90_dataset_windows_do_not_cross_demos(tmp_path: Path) -> None:
    _write_window_task(tmp_path / "task.hdf5", (12, 12, 3))
    dataset = Libero90Dataset(_window_cfg(tmp_path, horizon=10, val_horizon=5))

    assert len(dataset) == 6
    assert dataset[2]["action"][:, 0].tolist() == list(range(2, 12))
    assert dataset[3]["action"][:, 0].tolist() == list(range(100, 110))
    assert dataset[0]["obs"]["agentview_rgb"].shape == (10, 3, 128, 128)
    assert dataset[0]["action"].shape == (10, 7)


def test_libero90_validation_pads_with_terminal_frame(tmp_path: Path) -> None:
    _write_window_task(tmp_path / "task.hdf5", (12, 12, 3))
    validation = Libero90Dataset(
        _window_cfg(tmp_path, horizon=10, val_horizon=5)
    ).get_validation_dataset()
    sample = validation[0]

    assert len(validation) == 1
    assert sample["action"][:, 0].tolist() == [200, 201, 202, 202, 202]
    assert sample["goal"]["agentview_rgb"].equal(sample["obs"]["agentview_rgb"][2])
    assert sample["rel_stop_idx"].item() == 2


def test_libero90_dataset_rejects_empty_demo(tmp_path: Path) -> None:
    _write_window_task(tmp_path / "task.hdf5", (0, 1))

    with pytest.raises(ValueError, match="Demonstrations must contain at least one frame"):
        Libero90Dataset(_window_cfg(tmp_path, horizon=1, val_horizon=1))


def test_libero90_dataset_spawned_loader_orders_and_isolates_handles(
    tmp_path: Path,
) -> None:
    path = tmp_path / "task.hdf5"
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        for demo_idx, length in ((10, 1), (2, 2), (20, 1)):
            demo = data.create_group(f"demo_{demo_idx}")
            obs = demo.create_group("obs")
            for key in ("agentview_rgb", "eye_in_hand_rgb"):
                obs.create_dataset(
                    key,
                    data=np.full((length, 128, 128, 3), demo_idx, dtype=np.uint8),
                )
            demo.create_dataset(
                "actions",
                data=np.full((length, 7), demo_idx, dtype=np.float32),
            )
    cfg = OmegaConf.create(
        {
            "dataset_dir": str(tmp_path),
            "expected_tasks": 1,
            "val_demos_per_task": 1,
            "horizon": 1,
            "val_horizon": 1,
            "resolution": 128,
            "action_dim": 7,
            "obs_keys": ["agentview_rgb", "eye_in_hand_rgb"],
        }
    )
    dataset = Libero90Dataset(cfg)

    assert dataset[1]["action"][0, 0].item() == 2
    assert dataset[2]["action"][0, 0].item() == 10
    assert dataset._handles
    assert dataset.__getstate__()["_handles"] == {}
    actions = [
        batch["action"][0, 0].item()
        for batch in DataLoader(
            dataset, batch_size=None, num_workers=1, multiprocessing_context="spawn"
        )
    ]
    assert actions == [2, 2, 10]
