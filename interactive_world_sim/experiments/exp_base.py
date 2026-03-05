"""Base classes for experiments.

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research
template [repo](https://github.com/buoyancy99/research-template).
By its MIT license, you must keep the above sentence in `README.md`
and the `LICENSE` file to credit the author.
"""

import os
import pathlib
from abc import ABC
from typing import Dict, Optional, Union

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from omegaconf import DictConfig

from interactive_world_sim.utils.distributed_utils import is_rank_zero
from interactive_world_sim.utils.print_utils import cyan

torch.set_float32_matmul_precision("high")


class BaseExperiment(ABC):
    """Abstract class for an experiment.

    This generalizes the pytorch lightning Trainer & lightning Module to
    more flexible experiments that doesn't fit in the typical ml loop,
    e.g. multi-stage reinforcement learning benchmarks.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix # noqa
    compatible_algorithms: Dict = {}

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """Constructor

        Args:
            root_cfg: configuration file that contains everything about the experiment
            logger: a pytorch-lightning WandbLogger instance
            ckpt_path: an optional path to saved checkpoint
        """
        super().__init__()
        self.root_cfg = root_cfg
        self.cfg = root_cfg.experiment
        self.debug = root_cfg.debug
        self.logger = logger
        self.ckpt_path = ckpt_path
        self.algo = None

    def _build_algo(self) -> pl.LightningModule:
        """Build the lightning module

        :return:  a pytorch-lightning module to be launched
        """
        algo_name = self.root_cfg.algorithm._name  # noqa
        if algo_name not in self.compatible_algorithms:
            raise ValueError(
                f"Algorithm {algo_name} not found in compatible_algorithms for this Experiment class. "  # noqa
                "Make sure you define compatible_algorithms correctly and make sure that each key has "  # noqa
                "same name as yaml file under '[project_root]/configurations/algorithm' without .yaml suffix"  # noqa
            )
        if "ckpt_path" in self.root_cfg.algorithm:
            return self.compatible_algorithms[algo_name].load_from_checkpoint(
                self.root_cfg.algorithm.ckpt_path,
                cfg=self.root_cfg.algorithm,
                map_location="cuda:0",
            )
        else:
            return self.compatible_algorithms[algo_name](self.root_cfg.algorithm)

    def exec_task(self, task: str) -> None:
        """Executing a certain task stage specified by string.

        In most computer vision / nlp applications,
        tasks should be just train and test.
        In reinforcement learning,
        you might have more stages such as collecting dataset etc

        Args:
            task: a string specifying a task implemented for this experiment
        """
        if hasattr(self, task) and callable(getattr(self, task)):
            if is_rank_zero:
                print(cyan("Executing task:"), f"{task} out of {self.cfg.tasks}")
            getattr(self, task)()
        else:
            raise ValueError(
                f"Specified task '{task}' not defined for "
                f"class {self.__class__.__name__} or is not callable."
            )


class BaseLightningExperiment(BaseExperiment):
    """Abstract class for pytorch lightning experiments.

    Useful for computer vision & nlp where main components are
    simply models, datasets and train loop.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix # noqa
    compatible_algorithms: Dict = {}

    # each key has to be a yaml file under '[project_root]/configurations/dataset' without .yaml suffix # noqa
    compatible_datasets: Dict = {}

    def _build_training_loader(
        self,
    ) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        train_dataset = self._build_dataset("training")
        shuffle = (
            False
            if isinstance(train_dataset, torch.utils.data.IterableDataset)
            else self.cfg.training.data.shuffle
        )
        if train_dataset:
            return torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.training.data.num_workers),
                shuffle=shuffle,
                persistent_workers=False,
                pin_memory=False,
                prefetch_factor=1,
            )
        else:
            return None

    def _build_validation_loader(
        self,
    ) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        validation_dataset = self._build_dataset("validation")
        shuffle = (
            False
            if isinstance(validation_dataset, torch.utils.data.IterableDataset)
            else self.cfg.validation.data.shuffle
        )
        if validation_dataset is not None:
            return torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=self.cfg.validation.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.validation.data.num_workers),
                shuffle=shuffle,
                persistent_workers=False,
                pin_memory=False,
                prefetch_factor=1,
            )
        else:
            return None

    def _build_test_loader(
        self,
    ) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        test_dataset = self._build_dataset("test")
        shuffle = (
            False
            if isinstance(test_dataset, torch.utils.data.IterableDataset)
            else self.cfg.test.data.shuffle
        )
        if test_dataset:
            return torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.cfg.test.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.test.data.num_workers),
                shuffle=shuffle,
                persistent_workers=False,
                pin_memory=False,
                prefetch_factor=1,
            )
        else:
            return None

    def training(self) -> None:
        """All training happens here"""
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.training.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        if "checkpointing" in self.cfg.training:
            callbacks.append(
                ModelCheckpoint(
                    pathlib.Path(
                        hydra.core.hydra_config.HydraConfig.get()["runtime"][
                            "output_dir"
                        ]
                    )
                    / "checkpoints",
                    **self.cfg.training.checkpointing,
                )
            )

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger if self.logger else False,
            devices=self.cfg.num_devices,
            num_nodes=self.cfg.num_nodes,
            strategy=(
                DDPStrategy(find_unused_parameters=True)
                if torch.cuda.device_count() > 1
                else "auto"
            ),
            callbacks=callbacks,
            gradient_clip_val=self.cfg.training.optim.gradient_clip_val,
            val_check_interval=self.cfg.validation.val_every_n_step,
            limit_val_batches=self.cfg.validation.limit_batch,
            check_val_every_n_epoch=self.cfg.validation.val_every_n_epoch,
            accumulate_grad_batches=self.cfg.training.optim.accumulate_grad_batches,
            precision=self.cfg.training.precision,
            detect_anomaly=False,
            num_sanity_val_steps=int(self.cfg.debug),
            max_epochs=self.cfg.training.max_epochs,
            max_steps=self.cfg.training.max_steps,
            max_time=self.cfg.training.max_time,
            log_every_n_steps=self.cfg.training.log_every_n_steps,
        )

        train_dataloader = self._build_training_loader()
        val_dataloader = self._build_validation_loader()

        if hasattr(self.algo, "set_normalizer"):
            self.algo.set_normalizer(train_dataloader.dataset.get_normalizer())  # type: ignore

        trainer.fit(
            self.algo,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=self.ckpt_path,
        )

    def validation(self) -> None:
        """All validation happens here"""
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

        callbacks: list = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices=self.cfg.num_devices,
            num_nodes=self.cfg.num_nodes,
            strategy=(
                DDPStrategy(find_unused_parameters=True)
                if torch.cuda.device_count() > 1
                else "auto"
            ),
            callbacks=callbacks,
            limit_val_batches=self.cfg.validation.limit_batch,
            precision=self.cfg.validation.precision,
            detect_anomaly=False,
            inference_mode=self.cfg.validation.inference_mode,
        )

        val_dataloader = self._build_validation_loader()
        if hasattr(self.algo, "set_normalizer"):
            self.algo.set_normalizer(val_dataloader.dataset.get_normalizer())  # type: ignore
        trainer.validate(
            self.algo,
            dataloaders=val_dataloader,
            ckpt_path=self.ckpt_path,
        )

    def test(self) -> None:
        """All testing happens here"""
        if not self.algo:
            self.algo = self._build_algo()

        callbacks: list = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices=self.cfg.num_devices,
            num_nodes=self.cfg.num_nodes,
            strategy=(
                DDPStrategy(find_unused_parameters=True)
                if torch.cuda.device_count() > 1
                else "auto"
            ),
            callbacks=callbacks,
            limit_test_batches=self.cfg.test.limit_batch,
            precision=self.cfg.test.precision,
            detect_anomaly=False,
        )

        trainer.test(
            self.algo,
            dataloaders=self._build_test_loader(),
            ckpt_path=self.ckpt_path,
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
