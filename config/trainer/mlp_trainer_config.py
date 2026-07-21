from dataclasses import dataclass
from typing import Optional, Any

from omegaconf import MISSING

from models.mlp.mlp_trainer import train_and_evaluate, predict
from utils import fullname

@dataclass
class MLPTrainerConfig:
    _target_: str = fullname(train_and_evaluate)
    model: Any = MISSING
    train_dl: Any = MISSING
    val_dl: Any = MISSING
    optimizer: Any = MISSING
    loss_fn: Any = MISSING
    metric_fn: Any = MISSING
    num_epochs: int = 3
    save_dir: Optional[str] = MISSING
    writer: Optional[Any] = MISSING
    cuda: bool = False 


@dataclass
class MLPPredictConfig:
    _target_: str = fullname(predict)
    model: Any = MISSING
    pred_dl: Any = MISSING
    metric_fn: Any = MISSING
    cuda: bool = False 