from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Local copies of the hydra_ax_sweeper config dataclasses with
# safe `default_factory` usage to avoid import-time dataclass errors
@dataclass
class EarlyStopConfig:
    minimize: bool = True
    max_epochs_without_improvement: int = 10
    epsilon: float = 0.00001


@dataclass
class ExperimentConfig:
    name: Optional[str] = None
    objective_name: str = "train_acc"
    minimize: bool = True
    parameter_constraints: Optional[List[str]] = None
    outcome_constraints: Optional[List[str]] = None
    status_quo: Optional[Dict[str, Any]] = None


@dataclass
class ClientConfig:
    verbose_logging: bool = False
    random_seed: Optional[int] = 42


@dataclass
class AxConfig:
    max_trials: int = 10
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    params: Dict[str, Any] = field(default_factory=dict)
    is_noisy: bool = True


@dataclass
class AxSweeperConf:
    _target_: str = "hydra_plugins.hydra_ax_sweeper.ax_sweeper.AxSweeper"
    max_batch_size: Optional[int] = 5
    ax_config: AxConfig = field(default_factory=AxConfig)

