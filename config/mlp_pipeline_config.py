"""Config file for MKP grid hyperparameter searchers"""
from dataclasses import dataclass, field
from typing import List, Any, Dict, Union

from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from config.data.data_config import DataConfig
from config.launcher.launcher_config import SlurmConfig
from config.model.mlp_config import MLPLitModuleConfig
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
    optimized_metric: str = ("train/mae")
    loggers: Dict[str, Any] = field(default_factory=lambda: loggers)
    callbacks: Dict[str, Any] = field(default_factory=lambda: callbacks)

    test: bool = True
    load_testing_model: bool = False
    test_model_path: str = "/home/sabria/scratch_sabria/vpsc-hill/deep-learning-anisotropy/multirun/2026-07-06/13-34-19/0/outputs/trained_model.pt"
    test_results_file: str = "test_results.csv"
    test_results_dir: str = "./test_results"
    plot_test_results: bool = True
    test_results_plot: str = "test_results.png"

    predict: bool = True
    prediction_dataset_path : str = "/home/sabria/scratch_sabria/vpsc-hill/data/poly.csv"
    load_predictive_model: bool = True
    pred_model_path: str = "/home/sabria/scratch_sabria/vpsc-hill/deep-learning-anisotropy/multirun/2026-07-06/13-34-19/0/outputs/trained_model.pt"
    predictions_file: str = "predictions.csv"
    predictions_dir: str = "./predictions"
    plot_pred: bool = True
    predictions_plot: str = "predictions.png"

    datamodule: Any = MISSING
    model: Any = MISSING
    trainer: Any = MISSING

    launcher: Any = field(default_factory=SlurmConfig)

def register_configs():
    cs = ConfigStore()
    cs.store(group="datamodule", name="anisotropy", node=DataConfig)
    cs.store(group="trainer", name="cpu_trainer", node=CPUTrainerConfig)
    cs.store(group="model", name="mlp", node=MLPLitModuleConfig)
    cs.store(group="hydra/launcher", name="submitit_slurm_local", node=SlurmConfig)
    cs.store(name="mlp_hpo", node=HPOConfig)