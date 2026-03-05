from typing import Optional

import torch

from interactive_world_sim.algorithms.latent_dynamics import LatentWorldModel
from interactive_world_sim.datasets.latent_dynamics import (
    RealAlohaDataset,
    SimAlohaDataset,
)

from .exp_base import BaseLightningExperiment


class LatentDynExperiment(BaseLightningExperiment):
    """Latent dynamics experiment for interactive world simulation."""

    compatible_algorithms = dict(
        latent_world_model=LatentWorldModel,
    )

    compatible_datasets = dict(
        sim_aloha_dataset=SimAlohaDataset,
        real_aloha_dataset=RealAlohaDataset,
    )

    def _build_dataset(self, split: str) -> Optional[torch.utils.data.Dataset]:
        # build the dataset
        if not hasattr(self, "dataset"):
            self.dataset = self.compatible_datasets[
                self.root_cfg.dataset._name  # noqa
            ](self.root_cfg.dataset)
        if split == "training":
            return self.dataset
        elif split == "validation":
            return self.dataset.get_validation_dataset()
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")
