from typing import Any, Dict
from dataclasses import dataclass, field

from pyro.infer import SVI
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from pyro.optim import Adam
from omegaconf import MISSING

from models.bnn.network import PartialBNN, FullBNN
from utils import fullname

@dataclass
class PartialBNNConfig:
    _target_: str = fullname(PartialBNN)
    input_dim: int = 21
    output_dim: int = 6
    hidden_dim: int = 126
    n_layers: int = 4
    prior_scale: float = 0.01
    dataset_size: int = MISSING

@dataclass
class FullBNNConfig:
    _target_: str = fullname(FullBNN)
    input_dim: int = 21
    output_dim: int = 6
    hidden_dim: int = 252
    n_layers: int = 4
    prior_scale: float = 0.1
    dataset_size: int = MISSING

@dataclass
class GuideConfig:
    _target_: str = fullname(AutoLowRankMultivariateNormal)
    model: Any = MISSING
    rank: int = 100

@dataclass
class AdamOptimizerConfig:
    _target_: str = fullname(Adam)
    optim_args: Dict[str, Any] = field(
        default_factory=lambda: {"lr": 0.001, "betas": (0.95, 0.999)}
    )

@dataclass
class SVIConfig:
    _target_: str = fullname(SVI)
    model: Any = MISSING
    guide: Any = MISSING
    optim: Any = MISSING
    loss: Any = MISSING