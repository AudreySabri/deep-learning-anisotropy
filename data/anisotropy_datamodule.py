from pathlib import Path
from typing import Union, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import Subset, random_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, Normalizer

from data.csv_dataset import CSVDataset
from data.dataloaders import create_dataloader

class AnisotropyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_path: str,
        file_path: str,
        input_pattern: str,
        target_pattern: str,
        train_split: float = 0.8,
        val_split: float = 0.,
        test_split: float = 0.2,
        transform: Optional[str] = 'standard',
        batch_size: int = 256,
        num_workers: int = 0,
        cuda: bool = False,
        seed: int = 42
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.df_path = Path(f"{self.hparams.root_path}/{self.hparams.file_path}") 
        self.feature_names = None
        self.target_names = None

        self.transformers = {
            'standard': StandardScaler,
            'power': PowerTransformer,
            'normalize': Normalizer
        }
        self.input_scaler = None
        self.target_scaler = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None) -> None:
         
        if self.train_dataset is not None:
            return

        dataset = CSVDataset(
            df_path=self.df_path,
            input_pattern=self.hparams.input_pattern,
            target_pattern=self.hparams.target_pattern,
        )
        self.feature_names = dataset.get_feature_names()
        self.target_names = dataset.get_target_names()
        train_size = int(self.hparams.train_split * len(dataset))
        val_size = int(self.hparams.val_split * len(dataset))
        test_size = len(dataset) - train_size - val_size
        generator = torch.Generator().manual_seed(self.hparams.seed)
        indices = torch.randperm(len(dataset), generator=generator)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size]
        test_idx = indices[train_size+val_size:]
        if self.hparams.transform is not None:
            if self.hparams.transform not in self.transformers:
                raise ValueError(
                    f"Unknown transform: {self.hparams.transform}"
                )
            scaler_cls = self.transformers[self.hparams.transform]
            train_inputs = dataset.data[train_idx]
            self.input_scaler = scaler_cls()
            self.input_scaler.fit(train_inputs)
            dataset.data = self.input_scaler.transform(dataset.data)

            if dataset.targets is not None:
                train_targets = dataset.targets[train_idx]
                self.target_scaler = scaler_cls()
                self.target_scaler.fit(train_targets)
                dataset.targets = self.target_scaler.transform(dataset.targets)
        self.train_dataset = Subset(dataset, train_idx)
        self.val_dataset   = Subset(dataset, val_idx)
        self.test_dataset  = Subset(dataset, test_idx)

    def train_dataloader(self):
        return create_dataloader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=self.hparams.cuda
        )
    def val_dataloader(self):
        if len(self.val_dataset) > 0:
            return create_dataloader(
                dataset=self.val_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=False,
                pin_memory=self.hparams.cuda
            )
        else:
            return None
    def test_dataloader(self):
        if len(self.test_dataset) > 0:
            return create_dataloader(
                dataset=self.test_dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=False,
                pin_memory=self.hparams.cuda
            )
        else:
            return None
        
    @property
    def train_size(self):
        return len(self.train_dataset)
    
    @property
    def return_input_scaler(self):
        return self.input_scaler
    
    @property
    def return_target_scaler(self):
        if self.target_scaler is None:
            return None
        else:
            return self.target_scaler
    
    @property 
    def return_feature_names(self):
        return self.feature_names

    @property
    def return_target_names(self):
        return self.target_names
