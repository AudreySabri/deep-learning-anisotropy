from typing import Optional

import hydra
from omegaconf import DictConfig
from pyro.nn import PyroModule
from pytorch_lightning import LightningModule, Trainer, seed_everything

import utils

log = utils.get_logger(__name__)

def train(config: DictConfig) -> Optional[float]:
    """
    Contains an example training pipeline.
    Can additionally evaluate model on a testset, using best weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    if config.get("seed"):
        seed_everything(config.seed)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: dict = hydra.utils.instantiate(config.datamodule)

    log.info(f"Instantiating dataloader <{config.dataloader._target_}>")
    dataloader: dict = hydra.utils.instantiate(config.dataloader)

    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer)

    if config.get("train"):

        log.info("Starting training!")
        trainer.fit(model=model(dataset_size=len(dataloader)), train_dataloaders=dataloader)


