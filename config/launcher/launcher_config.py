from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf, LocalQueueConf

@dataclass
class LocalConfig(LocalQueueConf):
    """Configuration for submitit_local -- used for debugging only"""

    timeout_min: int = 60
    cpus_per_task: int = 6
   # gpus_per_node: int = 1
    tasks_per_node: int = 1
    mem_gb: int = 20
    nodes: int = 1

@dataclass
class SlurmConfig(SlurmQueueConf):
    """Configuration for submitit_slurm. These parameters can be overwritten in the cli with hydra.launcher.something_below=.."""

    partition: str = "cpu-dedicated"
    qos: str = "dedicated"
  #  gpus_per_node: int = 1
    tasks_per_node: int = 1
    cpus_per_task: int = 16
    mem_gb: int = 32
    nodes: int = 1
    timeout_min: int = 1200  # how long can the job run
    array_parallelism: int = 1  # how many jobs can run simultaneously
    # other options include:

    # qos: Optional[str] = None
    # comment: Optional[str] = None
    # constraint: Optional[str] = None
    # exclude: Optional[str] = None
    # Eg: {"mail-user": "blublu@fb.com", "mail-type": "BEGIN"}
    # additional_parameters: Dict[str, Any] = field(default_factory=lambda: add_parm)