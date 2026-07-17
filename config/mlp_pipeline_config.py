"""Config file for MKP grid hyperparameter searchers"""
from dataclasses import dataclass, field
from typing import List, Any, Dict, Union

from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from config.data.data_config import DataConfig
from config.launcher.launcher_config import SlurmConfig
from config.model.mlp_config import MLPConfig
from config.trainer.mlp_trainer_config import (
    CPUTrainerConfig,
    GPUTrainerConfig
)
from config.logging.logging_config import (
    TensorBoardLoggerConfig,
    TensorBoardCallbackConfig,
)

defaults: List[Union[str, Dict[str, str]]] = [
    "_self_",
    {"datamodule": "anisotropy"},
    {"trainer": "cpu_trainer"},
    {"model": "mlp"},
    {"override /hydra/launcher": "submitit_slurm_local"},
]

loggers = {"tensorboard": TensorBoardLoggerConfig}
callbacks = {"tensorboard_checkpoint": TensorBoardCallbackConfig}

@dataclass
class HPOConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    hydra: Any = field(default_factory=lambda: HydraConf())
    seed: int = 42

    train: bool = True
    optimized_metric: str = ("train_mae")
    loggers: Dict[str, Any] = field(default_factory=lambda: loggers)
    callbacks: Dict[str, Any] = field(default_factory=lambda: callbacks)

    predict: bool = True
    plot_pred: bool = True
    load_model: bool = True
    load_dir: str = "/home/sabria/scratch_sabria/vpsc-hill/deep-learning-anisotropy/multirun/2026-07-06/13-34-19/0/outputs"
    predictions_file: str = "predictions.csv"
    predictions_dir: str = "./predictions"
    predictions_plot: str = "predictions.png"

    datamodule: Any = MISSING
    model: Any = MISSING
    trainer: Any = MISSING

    launcher: Any = field(default_factory=SlurmConfig)

def register_configs():
    cs = ConfigStore()
    cs.store(group="datamodule", name="anisotropy", node=DataConfig)
    cs.store(group="trainer", name="cpu_trainer", node=CPUTrainerConfig)
    cs.store(group="model", name="partial_bnn", node=MLPConfig)
    cs.store(group="hydra/launcher", name="submitit_slurm_local", node=SlurmConfig)
    cs.store(name="hpo", node=HPOConfig)