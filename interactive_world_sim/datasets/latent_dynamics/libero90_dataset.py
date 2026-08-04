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
    """Lazy frame dataset for LIBERO task HDF5 files."""

    def __init__(self, cfg: DictConfig, split: str = "training") -> None:
        super().__init__()
        if split not in {"training", "validation"}:
            raise ValueError(f"Invalid split: {split}")

        self.cfg = cfg
        self.split = split
        self.resolution = cfg.resolution
        self.action_dim = cfg.action_dim
        self.obs_keys = list(cfg.obs_keys)
        self._handles: dict[Path, h5py.File] = {}

        task_paths = sorted(Path(cfg.dataset_dir).glob("*.hdf5"))
        if len(task_paths) != cfg.expected_tasks:
            raise ValueError(
                f"Expected {cfg.expected_tasks} task files, found {len(task_paths)}"
            )

        self._demos: list[tuple[Path, str, int]] = []
        self._frame_ends: list[int] = []
        frame_end = 0
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
                    frame_end += length
                    self._frame_ends.append(frame_end)

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
        for key in self.obs_keys:
            images = demo["obs"][key]
            if images.shape != (length, self.resolution, self.resolution, 3):
                raise ValueError(
                    f"Expected {key} images shaped "
                    f"({length}, {self.resolution}, {self.resolution}, 3)"
                )
        return length

    def __len__(self) -> int:
        return self._frame_ends[-1] if self._frame_ends else 0

    def _get_handle(self, path: Path) -> h5py.File:
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        demo_idx = bisect_right(self._frame_ends, idx)
        frame_idx = idx - (self._frame_ends[demo_idx - 1] if demo_idx else 0)
        path, demo_name, _ = self._demos[demo_idx]
        demo = self._get_handle(path)["data"][demo_name]
        obs = {
            key: torch.from_numpy(np.asarray(demo["obs"][key][frame_idx]))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            for key in self.obs_keys
        }
        return {
            "obs": obs,
            "goal": {key: value[0] for key, value in obs.items()},
            "action": torch.from_numpy(
                np.asarray(demo["actions"][frame_idx], dtype=np.float32)
            ).unsqueeze(0),
            "is_early_stop": torch.tensor([False]),
            "rel_stop_idx": torch.tensor([0]),
        }

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
