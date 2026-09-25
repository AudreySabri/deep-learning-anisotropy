"""Config file for MLP grid hyperparameter searchers"""
from dataclasses import dataclass, field
from typing import List, Any, Dict, Union, Optional

from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from config.data.data_config import DataConfig
from config.launcher.launcher_config import SlurmConfig
from config.model.mlp_config import (
    MLPConfig,
    MSELossConfig,
    TrackingMetricConfig,
    AdamOptimizerConfig,
)
from config.trainer.mlp_trainer_config import (
    MLPTrainerConfig,
    PredictorConfig,
)

defaults: List[Union[str, Dict[str, str]]] = [
    "_self_",
    {"datamodule": "anisotropy"},
    {"model": "mlp"},
    {"loss": "mse"},
    {"metric_fn": "mae"},
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

    train: bool = False
    save_dir: str = "./outputs"
    log_dir: str = "./logs"

    test: bool = False
    load_testing_model: bool = False
    test_model_path: Optional[str] = None
    test_results_dir: str = "./test_results"
    test_results_file: str = "test_results.csv"
    plot_test_results: bool = True
    test_results_plot: Optional[str] = "test_results.png"

    predict: bool = True
    prediction_dataset_path : Optional[str] = "/home/sabria/scratch_sabria/deep-learning-anisotropy/datasets/polynesia_db.csv"
    load_predictive_model: bool = True
    pred_model_path: Optional[str] = '/home/sabria/scratch_sabria/deep-learning-anisotropy/trained_models/trained_mlp_hill/outputs/best.pth.tar'
    predictions_dir: str = "./predictions"  # output directory for predictions
    predictions_file: str = "polynesia_predictions.csv" #output file for predictions
    plot_pred: bool = True
    predictions_plot: Optional[str] = "polynesia_predictions.png" #output file for predictions plot

    datamodule: Any = MISSING
    model: Any = MISSING
    loss: Any = MISSING
    metric_fn: Any = MISSING
    optimizer: Any = MISSING
    trainer: Any = MISSING
    predictor: Any = MISSING

    launcher: Any = field(default_factory=SlurmConfig)

def register_mlp_configs():
    """Register MLP-related dataclasses with Hydra `ConfigStore`.

    Registers the dataclass nodes for datamodule, model, loss, metric,
    optimizer, trainer, predictor and launcher so Hydra can instantiate them.
    """
    cs = ConfigStore()
    cs.store(group="datamodule", name="anisotropy", node=DataConfig)
    cs.store(group="model", name="mlp", node=MLPConfig)
    cs.store(group="loss", name="mse", node=MSELossConfig)
    cs.store(group="metric_fn", name="mae", node=TrackingMetricConfig)
    cs.store(group="optimizer", name="adam", node=AdamOptimizerConfig)
    cs.store(group="trainer", name="mlp_trainer", node=MLPTrainerConfig)
    cs.store(group="predictor", name="mlp_predictor", node=PredictorConfig)
    cs.store(group="hydra/launcher", name="submitit_slurm_local", node=SlurmConfig)
    cs.store(name="mlp", node=PipelineConfig)