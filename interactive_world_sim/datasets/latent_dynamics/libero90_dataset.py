from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from omegaconf import DictConfig

from interactive_world_sim.utils.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
    get_image_range_normalizer,
)

from .base_dataset import BaseImageDataset


class Libero90Dataset(BaseImageDataset):
    """Lazy temporal-window dataset for LIBERO task HDF5 files."""

    def __init__(self, cfg: DictConfig, split: str = "training") -> None:
        super().__init__()
        if split not in {"training", "validation"}:
            raise ValueError(f"Invalid split: {split}")

        self.cfg = cfg
        self.split = split
        self.resolution = cfg.resolution
        self.action_dim = cfg.action_dim
        self.obs_keys = list(cfg.obs_keys)
        self.horizon = cfg.horizon
        self.val_horizon = cfg.val_horizon
        self._handles: dict[Path, h5py.File] = {}

        task_paths = sorted(Path(cfg.dataset_dir).glob("*.hdf5"))
        if len(task_paths) != cfg.expected_tasks:
            raise ValueError(
                f"Expected {cfg.expected_tasks} task files, found {len(task_paths)}"
            )

        self._demos: list[tuple[Path, str, int]] = []
        self._sample_ends: list[int] = []
        sample_end = 0
        for path in task_paths:
            with h5py.File(path, "r") as file:
                demos = sorted(file["data"], key=self._demo_index)
                if len(demos) <= cfg.val_demos_per_task:
                    raise ValueError(
                        f"{path} has too few demonstrations for validation split"
                    )
                if split == "training":
                    demos = demos[: -cfg.val_demos_per_task]
                else:
                    demos = demos[-cfg.val_demos_per_task :]
                for demo_name in demos:
                    length = self._validate_demo(file["data"][demo_name])
                    self._demos.append((path, demo_name, length))
                    if split == "training":
                        sample_end += max(length - self.horizon + 1, 0)
                        self._sample_ends.append(sample_end)

        self.is_train = split == "training"
        self.is_val = split == "validation"

    @staticmethod
    def _demo_index(name: str) -> int:
        try:
            return int(name.rsplit("_", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"Demo name must end with an integer: {name}") from error

    def _validate_demo(self, demo: h5py.Group) -> int:
        actions = demo["actions"]
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"Expected actions with width {self.action_dim}")
        length = actions.shape[0]
        if length == 0:
            raise ValueError("Demonstrations must contain at least one frame")
        for key in self.obs_keys:
            images = demo["obs"][key]
            if images.shape != (length, self.resolution, self.resolution, 3):
                raise ValueError(
                    f"Expected {key} images shaped "
                    f"({length}, {self.resolution}, {self.resolution}, 3)"
                )
        return length

    def __len__(self) -> int:
        if self.split == "validation":
            return len(self._demos)
        return self._sample_ends[-1] if self._sample_ends else 0

    def _get_handle(self, path: Path) -> h5py.File:
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        if self.split == "training":
            demo_idx = bisect_right(self._sample_ends, idx)
            start = idx - (self._sample_ends[demo_idx - 1] if demo_idx else 0)
            real_length = self.horizon
        else:
            demo_idx = idx
            start = 0
            real_length = min(self._demos[demo_idx][2], self.val_horizon)
        path, demo_name, _ = self._demos[demo_idx]
        demo = self._get_handle(path)["data"][demo_name]
        obs = {
            key: torch.from_numpy(
                self._pad_terminal(
                    np.asarray(demo["obs"][key][start : start + real_length])
                )
            )
            .permute(0, 3, 1, 2)
            .float()
            .div(255.0)
            for key in self.obs_keys
        }
        actions = self._pad_terminal(
            np.asarray(demo["actions"][start : start + real_length], dtype=np.float32)
        )
        return {
            "obs": obs,
            "goal": {key: value[real_length - 1] for key, value in obs.items()},
            "action": torch.from_numpy(actions),
            "is_early_stop": torch.tensor([False]),
            "rel_stop_idx": torch.tensor([real_length - 1]),
        }

    def _pad_terminal(self, values: np.ndarray) -> np.ndarray:
        if self.split == "training" or len(values) == self.val_horizon:
            return values
        return np.concatenate(
            (values, np.repeat(values[-1:], self.val_horizon - len(values), axis=0))
        )

    def get_validation_dataset(self) -> Libero90Dataset:
        return Libero90Dataset(self.cfg, split="validation")

    def get_normalizer(
        self, mode: str = "none", **kwargs: dict
    ) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        for key in self.obs_keys:
            normalizer[key] = get_image_range_normalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            handle.close()
