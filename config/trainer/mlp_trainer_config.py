from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING
from pytorch_lightning.trainer import Trainer

from utils import fullname

@dataclass
class GPUTrainerConfig:
    _target_: str = fullname(Trainer)
    accelerator: str = "gpu"
    num_nodes: int = 1
    check_val_every_n_epoch: int = 1
    max_epochs: int = 2
    log_every_n_steps: int = 10
    fast_dev_run: bool = False


@dataclass
class CPUTrainerConfig:
    _target_: str = fullname(Trainer)
    accelerator: str = "cpu"
    check_val_every_n_epoch: int = 1
    max_epochs: int = 2
    log_every_n_steps: int = 10
    fast_dev_run: bool = False