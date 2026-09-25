from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING

from utils import fullname
from models.bnn.bnn_trainer import SVITrainer
from models.bnn.bnn_predict import BNNPredictor

@dataclass
class SVITrainerConfig:
    """Configuration for Stochastic Variational Inference (SVI) trainer."""
    _target_: str = fullname(SVITrainer)
    svi : Any = MISSING
    dataloader : Any = MISSING
    num_epochs : int = 50
    device : str = "cpu"  # or "cuda"   
    writer : Any = MISSING

@dataclass
class BNNPredictorConfig:
    """Configuration for Bayesian neural network predictor."""
    _target_: str = fullname(BNNPredictor)
    predictive : Any = MISSING
    dataloader : Any = MISSING
    device : str = "cpu"  # or "cuda"   
    scaler : Any = MISSING
     
