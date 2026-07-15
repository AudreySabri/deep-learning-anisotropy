from dataclasses import dataclass

from models.mlp import MLP
from utils import fullname

@dataclass
class MLPConfig:
    _target_: str = fullname(MLP)
    input_dim: int = 21
    output_dim: int = 6
    hidden_dim: int = 128
    dropout_rate: float = 0.5