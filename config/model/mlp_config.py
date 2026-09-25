from typing import Any
from dataclasses import dataclass

import torch
from omegaconf import MISSING

from models.mlp.network import MLP
from utils import fullname

@dataclass
class MLPConfig:
    _target_: str = fullname(MLP)
    input_dim: int = 21
    output_dim: int = 6
    hidden_dim: int = 252
    n_layers: int = 2
    dropout_rate: float = 0.2

@dataclass
class AdamOptimizerConfig:
    _target_: str = fullname(torch.optim.AdamW)
    params: Any = MISSING
    lr: float = 1e-4
    weight_decay: float = 5e-5


@dataclass
class MSELossConfig:
    _target_: str = fullname(torch.nn.MSELoss)

@dataclass
class TrackingMetricConfig:
    _target_: str = fullname(torch.nn.L1Loss)


