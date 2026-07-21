"""Config file for MKP grid hyperparameter searchers"""
from dataclasses import dataclass, field
from typing import List, Any, Dict, Union

from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch.nn import L1Loss

from config.data.data_config import DataConfig
from config.launcher.launcher_config import SlurmConfig
from config.model.mlp_config import (
    MLPConfig,
    MSELossConfig,
    AdamOptimizerConfig,
)
from config.trainer.mlp_trainer_config import (
    MLPTrainerConfig,
    PredictorConfig,
)
from utils import fullname

defaults: List[Union[str, Dict[str, str]]] = [
    "_self_",
    {"datamodule": "anisotropy"},
    {"model": "mlp"},
    {"loss": "mse"},
    {"optimizer": "adam"},
    {"trainer": "mlp_trainer"},
    {"predictor": "mlp_predictor"},
    {"override /hydra/launcher": "submitit_slurm_local"},
]

@dataclass
class PipelineConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    hydra: Any = field(default_factory=lambda: HydraConf())
    seed: int = 42

    train: bool = True
    tracking_metric: Any = fullname(L1Loss)
    save_dir: str = "./outputs"
    log_dir: str = "./logs"

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
    loss: Any = MISSING
    optimizer: Any = MISSING
    trainer: Any = MISSING
    predictor: Any = MISSING

    launcher: Any = field(default_factory=SlurmConfig)

def register_mlp_configs():
    cs = ConfigStore()
    cs.store(group="datamodule", name="anisotropy", node=DataConfig)
    cs.store(group="model", name="mlp", node=MLPConfig)
    cs.store(group="loss", name="mse", node=MSELossConfig)
    cs.store(group="optimizer", name="adam", node=AdamOptimizerConfig)
    cs.store(group="trainer", name="mlp_trainer", node=MLPTrainerConfig)
    cs.store(group="predictor", name="mlp_predictor", node=PredictorConfig)
    cs.store(group="hydra/launcher", name="submitit_slurm_local", node=SlurmConfig)
    cs.store(name="mlp", node=PipelineConfig)