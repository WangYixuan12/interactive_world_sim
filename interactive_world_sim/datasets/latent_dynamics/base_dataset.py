import copy
from typing import Dict

import torch
import torch.nn

from interactive_world_sim.utils.normalizer import LinearNormalizer
from interactive_world_sim.utils.sampler import SequenceSampler


class BaseLowdimDataset(torch.utils.data.Dataset):
    """Base class for low-dimensional datasets."""

    def get_validation_dataset(self) -> "BaseLowdimDataset":
        """Return a validation dataset."""
        # return an empty dataset by default
        return BaseLowdimDataset()

    def get_normalizer(self, mode: str, **kwargs: Dict) -> LinearNormalizer:
        """Return a normalizer."""
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        """Return all actions."""
        raise NotImplementedError()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a sample.

        output:
        obs: T, Do
        action: T, Da
        """
        raise NotImplementedError()


class BaseImageDataset(torch.utils.data.Dataset):
    """Base class for image datasets."""

    def __init__(self) -> None:
        super().__init__()

        self.is_train = True
        self.is_val = False
        self.is_test = False

        self.train_mask = None
        self.val_mask = None
        self.test_mask = None

    def get_normalizer(self, mode: str, **kwargs: Dict) -> LinearNormalizer:
        """Return a normalizer."""
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        """Return all actions."""
        raise NotImplementedError()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a sample.

        output:
        obs:
            key: T, *
        action: T, Da
        """
        raise NotImplementedError()

    def get_validation_dataset(self) -> "BaseImageDataset":
        """Return a validation dataset."""
        val_set = copy.copy(self)
        val_set.is_val = True
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.val_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
            skip_idx=self.skip_idx,
            goal_sample=self.goal_sample,
            skip_frame=self.skip_frame,
        )
        val_set.train_mask = self.val_mask
        return val_set

    def get_test_dataset(self) -> "BaseImageDataset":
        """Return a test dataset."""
        test_set = copy.copy(self)
        test_set.is_test = True
        test_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.test_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.test_mask,
            skip_idx=self.skip_idx,
        )
        test_set.train_mask = self.test_mask
        return test_set
