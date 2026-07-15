"""Config file for Bayesian hyperparameter searchers -- uses advanced sweeper setup"""
from dataclasses import dataclass, field
from typing import List, Any, Dict, Union

from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from config.data.data_config import DataConfig
from config.launcher.launcher_config import SlurmConfig
from config.logging.logging_config import (
    TensorBoardLoggerConfig,
    TensorBoardCallbackConfig,
)
from config.model.bnn_config import (
    PartialBNNConfig, 
    GuideConfig, 
    ELBOLossConfig,
    SVIConfig,
)

defaults: List[Union[str, Dict[str, str]]] = [
    "_self_",
    {"datamodule": "anisotropy"},
    {"trainer": "cpu_trainer"},
    {"model": "partial_bnn"},
    {"guide": "low_rank"},
    {"loss_fn": "elbo"},
    {"lightning": "pyro_lightning_module"},
    {"override /hydra/launcher": "submitit_slurm_local"},
]

loggers = {"tensorboard": TensorBoardLoggerConfig}
callbacks = {"tensorboard_checkpoint": TensorBoardCallbackConfig}

@dataclass
class HPOConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    hydra: Any = field(default_factory=lambda: HydraConf(launcher=SlurmConfig(), sweeper={}))
    seed: int = 42
    train: bool = True
    optimized_metric: str = ("train_acc")
    loggers: Dict[str, Any] = field(default_factory=lambda: loggers)
    callbacks: Dict[str, Any] = field(default_factory=lambda: callbacks)

    datamodule: Any = MISSING
    model: Any = MISSING
    guide: Any = MISSING
    loss_fn: Any = MISSING
    lightning: Any = MISSING
    trainer: Any = MISSING

    launcher: Any = field(default_factory=SlurmConfig)

def register_configs():
    cs = ConfigStore()
    cs.store(group="datamodule", name="anisotropy", node=DataConfig)
    cs.store(group="trainer", name="cpu_trainer", node=TrainerConfig)
    cs.store(group="model", name="partial_bnn", node=PartialBNNConfig)
   # cs.store(group="model", name="full", node=FullBNNConfig)
    cs.store(group="guide", name="low_rank", node=GuideConfig)
    cs.store(group="loss_fn", name="elbo", node=ELBOLossConfig)
    cs.store(group="lightning", name="pyro_lightning_module", node=PyroLightningConfig)
    cs.store(group="hydra/launcher", name="submitit_slurm_local", node=SlurmConfig)
    cs.store(name="hpo", node=HPOConfig)