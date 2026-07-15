"""Collection of tensorboard utilities to enhance the tensorboard logging"""
import logging
import shutil
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only, model_summary

from utils import get_logger

log = get_logger(__name__)
log.setLevel(logging.INFO)


class TensorBoardModelCheckpoint(ModelCheckpoint):
    """
    This class wraps ModelCheckpoint to allow the storing of checkpoints and model summaries to TensorBoard instead
    """

    def __init__(
        self,
        dirpath: str,
        del_ckpts_outside_tensorboard: bool,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        dirpath: str
            the directory where the tensorboard logs will be stored
        del_ckpts_outside_tensorboard: bool
            if True, we delete the 'normal' Lightning checkpoints after storing them in TensorBoard,
            this removes the redundancy and saves space
        args
        kwargs
        """
        super().__init__(*args, **kwargs)
        self.del_ckpts_outside_tensorboard = del_ckpts_outside_tensorboard
        self.dirpath = dirpath

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_train_start(trainer, pl_module)
        self.store_model_summary(trainer, pl_module)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        self.store_model_summary(trainer, pl_module)

    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_train_end(trainer, pl_module) 
        # save all the checkpoints that can be found in the checkpoint folder to tensorboard
        self.store_models()
    
    @rank_zero_only
    def store_models(self) -> None:
        """
        Optionally delete duplicate Lightning checkpoints
        after they are safely written.
        """

        if not self.del_ckpts_outside_tensorboard:
            return

        if self.dirpath is None:
            return

        checkpoint_dir = Path(self.dirpath)

        if checkpoint_dir.exists():
            log.info(f"Deleting duplicate checkpoints at {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)


    @rank_zero_only
    def store_model_summary(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Stores a summary of the model as in tensorboard
        Parameters
        ----------
        trainer: Trainer
            the trainer instance
        pl_module: LightningModule
            the lightning module of which a summary is to be stored

        Returns
        -------

        """
        summary = str(model_summary.summarize(pl_module, max_depth=10))
        log_dir = Path(trainer.log_dir) if trainer.log_dir is not None else Path(self.dirpath)
        log_dir.mkdir(parents=True, exist_ok=True)

        summary_file = log_dir / "model_summary.txt"
        with open(summary_file, "w") as f:
            f.write(summary)
        log.info(f"Saved model summary to {summary_file}")