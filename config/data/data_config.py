from typing import Optional
from dataclasses import dataclass, field

from data.anisotropy_datamodule import AnisotropyDataModule
from utils import fullname

@dataclass
class DataConfig:
    """Configuration for loading and preprocessing data."""
    _target_: str = fullname(AnisotropyDataModule)
    root_path: str = '/home/sabria/scratch_sabria/vpsc-hill/data'
    file_path: str = 'aug_db.csv'
    input_pattern: str = r'^c\d+_out$'
    target_pattern: str = r'^hill\d+_out$'
 #   target_pattern: str = r'^q\d+$'
    train_split: float = 0.75
    val_split: float = 0.20
    test_split: float = 0.05
    transform: Optional[str] = 'standard'
    batch_size: int = 1024
    num_workers: int = 1
    cuda: bool = False
    seed: int = 42
