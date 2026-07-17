from dataclasses import dataclass, field
from typing import Optional, List

from pytorch_lightning.loggers import TensorBoardLogger
from loggers.tensorboard_utils import TensorBoardModelCheckpoint
from utils import fullname


@dataclass
class TensorBoardLoggerConfig:
    _target_: str = fullname(TensorBoardLogger)
    save_dir: str = "./tensorboard_runs"
    name: str = "mlp_experiment"
    version: Optional[str] = None
    log_graph: bool = True


@dataclass
class TensorBoardCallbackConfig:
    _target_: str = fullname(TensorBoardModelCheckpoint)
    del_ckpts_outside_tensorboard: bool = True
    dirpath: str = "checkpoints"
    mode: str = "min"
    save_on_train_epoch_end: bool = True
    save_top_k: int = 1
    save_weights_only: bool = False
    every_n_epochs: int = 1
    monitor:  Optional[str] = "train_mae"
    verbose: bool = True