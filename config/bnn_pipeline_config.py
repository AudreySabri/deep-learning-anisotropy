"""Config file for BNN grid hyperparameter searchers"""
from dataclasses import dataclass, field
from typing import List, Any, Dict, Union, Optional

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
    {"model": "partial_bnn"},
    {"guide": "low_rank"},
    {"optimizer": "adam"},
    {"inference": "svi"},
    {"trainer": "cpu"},
    {"predictor": "bnn_predictor"},
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
    trained_model: str = "trained_bnn_model.pt"
    param_store: str = "bnn_param_store.pt"

    test: bool = False
    load_testing_model: bool = False
    test_model_path: Optional[str] = None
    test_param_store_path: Optional[str] = None
    test_num_samples: int = 20
    test_results_dir: str = "./test_results"
    test_results_file: str = "test_results.csv"
    plot_test_results: bool = True
    test_results_plot: Optional[str] = "test_results.png"

    predict: bool = True
    prediction_dataset_path : Optional[str] = "/home/sabria/scratch_sabria/deep-learning-anisotropy/datasets/carpathes_db.csv"
    load_predictive_model: Optional[bool] = True
    pred_model_path: Optional[str] = "/home/sabria/scratch_sabria/deep-learning-anisotropy/trained_models/trained_bnn_hill/outputs/trained_bnn_model.pt"
    pred_param_store_path: Optional[str] = "/home/sabria/scratch_sabria/deep-learning-anisotropy/trained_models/trained_bnn_hill/outputs/bnn_param_store.pt"
    pred_num_samples: int = 20
    predictions_dir: str = "./predictions" # output directory for predictions
    predictions_file: str = "carpathes_predictions.csv" #output file for predictions
    plot_pred: bool = True
    predictions_plot: Optional[str] = "carpathes_predictions.png" #output file for predictions plot

    datamodule: Any = MISSING
    model: Any = MISSING
    guide: Any = MISSING
    optimizer: Any = MISSING
    inference: Any = MISSING
    trainer: Any = MISSING
    predictor: Any = MISSING

    launcher: Any = field(default_factory=SlurmConfig)

def register_bnn_configs():
    """Register BNN-related dataclasses with Hydra `ConfigStore`.

    This function binds the dataclass config nodes used by the BNN pipeline
    (datamodule, model, guide, optimizer, inference, trainer, predictor,
    launcher) so they are discoverable by Hydra at runtime.
    """
    cs = ConfigStore()
    cs.store(group="datamodule", name="anisotropy", node=DataConfig)
    cs.store(group="model", name="partial_bnn", node=PartialBNNConfig)
   # cs.store(group="model", name="full", node=FullBNNConfig)
    cs.store(group="guide", name="low_rank", node=GuideConfig)
    cs.store(group="optimizer", name="adam", node=AdamOptimizerConfig)
    cs.store(group="inference", name="svi", node=SVIConfig)
    cs.store(group="trainer", name="cpu", node=SVITrainerConfig)
    cs.store(group="predictor", name="bnn_predictor", node=BNNPredictorConfig)
    cs.store(group="hydra/launcher", name="submitit_slurm_local", node=SlurmConfig)
    cs.store(name="bnn", node=PipelineConfig)