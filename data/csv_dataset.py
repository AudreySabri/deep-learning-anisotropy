"""Torch Dataset definition for loading anisotropy data from csv files."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, df_path, input_pattern=None, target_pattern=None):
        """
        Extract data from csv columns and store them for __getitem__ to access.

        Args:
            df_path: (string) path to the csv file containing the dataset
            input_pattern: (string) regex pattern to match input column names
            target_pattern: (string) regex pattern to match target column names
        """
        df_sample = pd.read_csv(df_path, nrows=1)

        self.data_columns = list(df_sample.filter(regex=input_pattern).columns)
        if not self.data_columns:
            raise ValueError(f"No columns matched input pattern: {input_pattern}")

        if target_pattern is not None:
            self.pred_columns = list(df_sample.filter(regex=target_pattern).columns)
            if not self.pred_columns:
                raise ValueError(f"No columns matched target pattern: {target_pattern}")
        else:
            self.pred_columns = []

        usecols = self.data_columns + self.pred_columns
        dtype_map = {col: np.float32 for col in usecols}

        self.df = pd.read_csv(df_path, usecols=usecols, dtype=dtype_map)

        self.data = self.df[self.data_columns].to_numpy(copy=False)
        self.targets = (
            self.df[self.pred_columns].to_numpy(copy=False)
            if self.pred_columns else None
        )
    def __len__(self):
        # return size of dataset
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Fetch input data and prediction target from dataset at index idx.

        Args:
            idx: (int) index of the sample to fetch
            
        Returns:
            data: (Tensor) input data corresponding to input columns
            target: (Tensor) prediction target corresponding to pred columns
        """
        data = self.data[idx]
        data_tensor = torch.from_numpy(data)
        
        if self.targets is not None:
            target = self.targets[idx]
            target_tensor = torch.from_numpy(target)
            return data_tensor, target_tensor
        else:
            return data_tensor 
        
    def get_feature_names(self):
        """Get the names of input features."""
        return self.data_columns
    
    def get_target_names(self):
        """Get the names of target/output features."""
        if self.pred_columns is not None:
            return self.pred_columns
        
    def get_feature_dim(self):
        """Get the dimensionality of input features."""
        return len(self.data_columns)
    
    def get_target_dim(self):
        """Get the dimensionality of target/output."""
        if self.pred_columns is not None:
            return len(self.pred_columns)
    