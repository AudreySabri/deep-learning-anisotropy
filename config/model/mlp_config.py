from dataclasses import dataclass

import torch

from models.mlp.network import MLP
from utils import fullname

@dataclass
class MLPConfig:
    _target_: str = fullname(MLP)
    input_dim: int = 21
    output_dim: int = 6
    hidden_dim: int = 128
    dropout_rate: float = 0.5

@dataclass
class AdamOptimizerConfig:
    _target_: str = fullname(torch.optim.Adam)
    lr: float = 0.001
    weight_decay: float = 0.0005


@dataclass
class MSELossConfig:
    _target_: str = fullname(torch.nn.MSELoss)


