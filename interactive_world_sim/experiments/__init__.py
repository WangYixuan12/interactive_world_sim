import pathlib
from typing import Optional, Union

from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig

from .exp_base import BaseExperiment
from .exp_latent_dyn import LatentDynExperiment

# each key has to be a yaml file under '[project_root]/configurations/experiment' without .yaml suffix # noqa
exp_registry = dict(
    exp_latent_dyn=LatentDynExperiment,
)


def build_experiment(
    cfg: DictConfig,
    logger: Optional[WandbLogger] = None,
    ckpt_path: Optional[Union[str, pathlib.Path]] = None,
) -> BaseExperiment:
    """Build an experiment instance based on registry

    :param cfg: configuration file
    :param logger: optional logger for the experiment
    :param ckpt_path: optional checkpoint path for saving and loading
    :return:
    """
    if cfg.experiment._name not in exp_registry:  # noqa
        raise ValueError(
            f"Experiment {cfg.experiment._name} not found in registry {list(exp_registry.keys())}. "  # noqa
            "Make sure you register it correctly in 'interactive_world_sim/experiments/__init__.py' "
            "under the same name as yaml file."
        )

    return exp_registry[cfg.experiment._name](cfg, logger, ckpt_path)  # noqa
