from pathlib import Path
from typing import Union, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import Subset, random_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, Normalizer

from data.csv_dataset import CSVDataset
from data.dataloaders import create_dataloader

class AnisotropyDataModule(pl.LightningDataModule):
    """Data module for handling anisotropy dataset."""
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
        """
        Args:
            root_path: (str) path to the root directory containing the dataset
            file_path: (str) path to the csv file containing the dataset
            input_pattern: (str) regex pattern to match input column names
            target_pattern: (str) regex pattern to match target column names
            train_split: (float) fraction of dataset to use for training
            val_split: (float) fraction of dataset to use for validation
            test_split: (float) fraction of dataset to use for testing
            transform: (Optional[str]) type of transformation to apply to the data
            batch_size: (int) number of samples per batch
            num_workers: (int) number of subprocesses to use for data loading
            cuda: (bool) whether to use CUDA for data loading
            seed: (int) random seed for reproducibility
        """
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
        """Prepare datasets and scalers for training/validation/testing.

        Splits the CSV-backed `CSVDataset` into train/val/test subsets, fits
        the requested input/target scalers on the training split, and stores
        the resulting `Subset` objects on the instance.

        Args:
            stage (Optional[str]): Optional stage name (unused).
        """
        
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
        """Create a DataLoader for the training subset.

        Returns a PyTorch `DataLoader` configured with the module's batch
        size, number of workers and pin_memory settings.
        """
        return create_dataloader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=self.hparams.cuda
        )
    def val_dataloader(self):
        """Create a DataLoader for the validation subset or return `None`.

        Returns a `DataLoader` when the validation split is non-empty, else
        returns `None`.
        """
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
        """Create a DataLoader for the test subset or return `None`.

        Returns a `DataLoader` when the test split is non-empty, else returns
        `None`.
        """
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
        """Return the number of samples in the training subset."""
        return len(self.train_dataset)
    
    @property
    def return_input_scaler(self):
        """Return the fitted input scaler (or `None`)."""
        return self.input_scaler
    
    @property
    def return_target_scaler(self):
        """Return the fitted target scaler (or `None`)."""
        if self.target_scaler is None:
            return None
        else:
            return self.target_scaler
    
    @property 
    def return_feature_names(self):
        """Return list of input feature names extracted from the CSV."""
        return self.feature_names

    @property
    def return_target_names(self):
        """Return list of target feature names extracted from the CSV."""
        return self.target_names
