"""Config file for grid hyperparameter searchers"""
from dataclasses import dataclass, field
from typing import List, Any, Dict, Union

from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from config.data.data_config import DataConfig
from config.trainer.bnn_trainer_config import (
    SVITrainerConfig,
    BNNPredictorConfig
)
from config.launcher.launcher_config import SlurmConfig
from config.model.bnn_config import (
    PartialBNNConfig, 
    GuideConfig, 
    AdamOptimizerConfig,
    SVIConfig,
)

defaults: List[Union[str, Dict[str, str]]] = [
    "_self_",
    {"datamodule": "anisotropy"},
    {"trainer": "cpu"},
    {"model": "partial_bnn"},
    {"guide": "low_rank"},
    {"optimizer": "adam"},
    {"inference": "svi"},
    {"predictor": "bnn_predictor"},
    {"override /hydra/launcher": "submitit_slurm_local"},
]

@dataclass
class HPOConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    hydra: Any = field(default_factory=lambda: HydraConf())
    seed: int = 42

    train: bool = False
    save_dir: str = "./outputs"
    log_dir: str = "./logs"
    trained_model: str = "trained_model.pt"
    param_store: str = "param_store.pt"

    predict: bool = True
    plot_pred: bool = True
    load_model: bool = True
    load_dir: str = "/home/sabria/scratch_sabria/vpsc-hill/deep-learning-anisotropy/multirun/2026-07-06/13-34-19/0/outputs"
    num_samples: int = 20
    predictions_file: str = "predictions.csv"
    predictions_dir: str = "./predictions"
    predictions_plot: str = "predictions.png"

    datamodule: Any = MISSING
    trainer: Any = MISSING
    model: Any = MISSING
    guide: Any = MISSING
    optimizer: Any = MISSING
    inference: Any = MISSING
    predictor: Any = MISSING

    launcher: Any = field(default_factory=SlurmConfig)

def register_configs():
    cs = ConfigStore()
    cs.store(group="datamodule", name="anisotropy", node=DataConfig)
    cs.store(group="trainer", name="cpu", node=SVITrainerConfig)
    cs.store(group="model", name="partial_bnn", node=PartialBNNConfig)
   # cs.store(group="model", name="full", node=FullBNNConfig)
    cs.store(group="guide", name="low_rank", node=GuideConfig)
    cs.store(group="optimizer", name="adam", node=AdamOptimizerConfig)
    cs.store(group="inference", name="svi", node=SVIConfig)
    cs.store(group="predictor", name="bnn_predictor", node=BNNPredictorConfig)
    cs.store(group="hydra/launcher", name="submitit_slurm_local", node=SlurmConfig)
    cs.store(name="hpo", node=HPOConfig)