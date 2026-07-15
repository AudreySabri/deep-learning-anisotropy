import logging
from pathlib import Path
from time import perf_counter
from typing import Optional

import hydra
import torch
import pyro
from omegaconf import DictConfig
from pyro.infer import Trace_ELBO
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import (
    LightningDataModule,
    seed_everything,
)

import utils
from utils.plotting import plot_bnn_predictions

log = utils.get_logger(__name__)

def train(config: DictConfig) -> Optional[float]:
    """
    Contains an example training pipeline.
    Can additionally evaluate model on a testset, using best weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    """

    if config.get("seed"):
        seed_everything(config.seed)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()
    train_dl = datamodule.train_dataloader()

    log.info(f"Instantiating model <{config.model._target_}>")
   # 1. instantiate BNN
    model = hydra.utils.instantiate(
        config.model,
        dataset_size=datamodule.train_size
    )


    log.info(f"Model instantiated with {sum(p.numel() for p in model.parameters())} parameters")
    # 2. guide
    guide = hydra.utils.instantiate(
        config.guide,
        model=model
    )
    log.info(f"Guide instantiated with {sum(p.numel() for p in guide.parameters())} parameters")
    # 3. loss
    loss = Trace_ELBO()
    log.info(f"Loss instantiated")
    # 4. svi
    # instantiate optimizer
    optimizer = hydra.utils.instantiate(config.optimizer, _convert_="all")

    svi = hydra.utils.instantiate(
        config.inference,
        model=model,
        guide=guide,
        loss=loss,
        optim=optimizer,
    )
    log.info(f"Inference algorithm instantiated")

    log.info(f"Instantiating Tensorboard logger")
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=config.log_dir)

    if config.get("train"):
        log.info(f"Starting training <{config.trainer._target_}>")
        hydra.utils.call(
            config.trainer,
            svi=svi,
            dataloader=train_dl,
            num_epochs=config.trainer.num_epochs,
            device=config.trainer.device,
            writer=writer,
        )

        log.info(f"Training completed")
        log.info("Saving model and guide")

        save_dir = Path(config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model": model.state_dict(),
            "guide": guide
            }, 
            save_dir / config.trained_model,
            )
        pyro.get_param_store().save(save_dir / config.param_store)
    
    if config.get("predict"):
        log.info(f"Starting prediction")
        start_mem = utils._get_memory_usage_mb()

        if config.get("load_model"):
            log.info(f"Loading model and guide from disk")
            load_dir = Path(config.load_dir)
            checkpoint = torch.load(load_dir / config.trained_model, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            guide = checkpoint["guide"]
            pyro.get_param_store().load(load_dir / config.param_store)

        predictive = pyro.infer.Predictive(model, 
                                           guide=guide, 
                                           num_samples=config.get("num_samples"),
                                           return_sites=["obs"],
        )
        test_dl = datamodule.test_dataloader()
        target_features = datamodule.return_target_names
        predictions, pred_df = hydra.utils.call(
            config.predictor,
            predictive=predictive,
            dataloader=test_dl,
            device=config.predictor.device,
            scaler=datamodule.return_target_scaler,
        )

        end_mem = utils._get_memory_usage_mb()

        log.info(f"Prediction completed")
        log.info("Memory usage: %.2f MB" % (end_mem - start_mem))

        log.info("Saving predictions to disk")
        pred_dir = Path(config.get("predictions_dir"))
        pred_dir.mkdir(parents=True, exist_ok=True)
        if config.get("plot_pred"):
            plot_bnn_predictions(
                results_dict=predictions,
                target_features=datamodule.return_target_names,
                save_dir=pred_dir,
                filename=config.get("predictions_plot")
            )
        pred_df.to_csv(pred_dir / config.get("predictions_file"), index=False)
    