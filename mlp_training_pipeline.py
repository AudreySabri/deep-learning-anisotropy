import time
from pathlib import Path
from typing import List, Optional

import torch
import hydra
import pandas as pd
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    seed_everything,
)
from torch.utils.tensorboard import SummaryWriter

import utils
from utils.plotting import plot_mlp_predictions


log = utils.get_logger(__name__)

# pylint: disable = protected-access
def train_mlp(config: DictConfig) -> Optional[float]:
    """
    Training pipeline.
    Can additionally evaluate model on a testset, using best weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed)

    # Init datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()
    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()

    # Init model
    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(
        config.model,
    )
    log.info(f"Model instantiated with {sum(p.numel() for p in model.parameters())} parameters")

    # Init loss
    loss = hydra.utils.instantiate(
        config.loss,
    )
    metric_fn = hydra.utils.instantiate(
        config.metric_fn,
    )
    log.info(f"Loss and tracking metric instantiated")

    # Init optimizer
    optimizer = hydra.utils.instantiate(
        config.optimizer,
        params= model.parameters()
    )
    log.info(f"Optimizer instantiated")

    # Init loggers and out dir
    log.info(f"Instantiating Tensorboard logger")
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=config.log_dir)

    # Train the model
    if config.get("train"):
        log.info("Starting training!")
        hydra.utils.call(
            config.trainer,
            model=model,
            train_dl=train_dl,
            val_dl=val_dl,
            optimizer=optimizer,
            loss_fn=loss,
            metric_fn=metric_fn,
            num_epochs=config.trainer.num_epochs,
            save_dir=config.save_dir,
            writer=writer,
            cuda=config.trainer.cuda,
        )
        log.info(f"Training completed")

    if config.get("test"):
        log.info("Starting prediction on test set!")
        
        start = time.time()
        start_mem = utils._get_memory_usage_mb()
        if config.get("load_testing_model"):
            log.info(f"Loading model from disk")
            checkpoint = torch.load(config.test_model_path, weights_only=False)
            state = checkpoint.state_dict()
            model.load_state_dict(state)
        test_dl = datamodule.test_dataloader()
        predictions, ground_truth = hydra.utils.call(
            config.predictor,
            model=model,
            pred_dl=test_dl,
            metric_fn=metric_fn,
            cuda=config.predictor.cuda,
        )
        end = time.time()
        end_mem = utils._get_memory_usage_mb()

        log.info("Prediction time: %.2f seconds", end - start)
        log.info("Memory usage: %.2f MB" % (end_mem - start_mem))

        log.info("Saving predictions to disk")
        target_scaler=datamodule.return_target_scaler
        predictions = target_scaler.inverse_transform(predictions)
        ground_truth= target_scaler.inverse_transform(ground_truth)

        test_results = {}
        target_features=datamodule.return_target_names
        for i, feature_name in enumerate(target_features):
            test_results [f'ground_truth_{feature_name}'] = ground_truth[:, i]
            test_results [f'predictions_{feature_name}'] = predictions[:, i]
        test_df = pd.DataFrame(test_results)
        test_dir = Path(config.get("test_results_dir"))
        test_dir.mkdir(parents=True, exist_ok=True)
        if config.get("plot_test_results"):
            plot_mlp_predictions(
                results_dict=test_results,
                target_features=datamodule.return_target_names,
                save_dir=test_dir,
                filename=config.get("test_results_plot")
            )
        test_df.to_csv(test_dir/ config.get("test_results_file"), index=False)
    
    if config.get("predict"):
        from data.csv_dataset import CSVDataset
        from data.dataloaders import create_dataloader

        log.info(f"Starting prediction on specified dataset!")
        start = time.time()
        start_mem = utils._get_memory_usage_mb()

        pred_db = config.get("prediction_dataset_path")
        pred_dataset = CSVDataset(
                        df_path=pred_db,
                        input_pattern=config.datamodule.input_pattern,
                        target_pattern=config.datamodule.target_pattern,
                    )
        input_scaler = datamodule.return_input_scaler
        target_scaler = datamodule.return_target_scaler
        pred_dataset.data = input_scaler.transform(pred_dataset.data)
        pred_dataset.targets = target_scaler.transform(pred_dataset.targets)
        pred_dl = create_dataloader(
                            dataset=pred_dataset,
                            batch_size=config.datamodule.batch_size,
                            num_workers=config.datamodule.num_workers,
                            shuffle=False,
                            pin_memory=config.datamodule.cuda,
                        )

        if config.get("load_predictive_model"):
            log.info(f"Loading model from disk")
            checkpoint = torch.load(config.pred_model_path, weights_only=False)
            state = checkpoint.state_dict()
            model.load_state_dict(state)
        predictions, ground_truth = hydra.utils.call(
            config.predictor,
            model=model,
            pred_dl=pred_dl,
            metric_fn=metric_fn,
            cuda=config.predictor.cuda,
        )

        end = time.time()
        end_mem = utils._get_memory_usage_mb()
        log.info("Prediction time: %.2f seconds", end - start)
        log.info("Memory usage: %.2f MB" % (end_mem - start_mem))

        log.info("Saving predictions to disk")
        predictions = target_scaler.inverse_transform(predictions)
        ground_truth= target_scaler.inverse_transform(ground_truth)

        pred_results = {}
        target_features=datamodule.return_target_names
        for i, feature_name in enumerate(target_features):
            pred_results[f'ground_truth_{feature_name}'] = ground_truth[:, i]
            pred_results[f'predictions_{feature_name}'] = predictions[:, i]
        pred_df = pd.DataFrame(pred_results)
        pred_dir = Path(config.get("predictions_dir"))
        pred_dir.mkdir(parents=True, exist_ok=True)
        if config.get("plot_pred"):
            plot_mlp_predictions(
                results_dict=pred_results,
                target_features=target_features,
                save_dir=pred_dir,
                filename=config.get("predictions_plot")
            )
        pred_df.to_csv(pred_dir/ config.get("predictions_file"), index=False)
    
    log.info("Finalizing!")
