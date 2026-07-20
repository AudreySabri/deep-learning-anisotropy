"""The lightning trainer module for our MLP."""
from typing import Any, List

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric
from torchmetrics.regression import MeanAbsoluteError

# pylint: disable = abstract-method
class MLPLitModule(LightningModule):
    """
    LightningModule for MLP Regression.

    """

    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn,
        optimizer_config: DictConfig,
    ):
        super().__init__()

        # ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = self.hparams.net
        self.optimizer_config = self.hparams.optimizer_config

        # loss function
        self.criterion = self.hparams.loss

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        # for logging best so far validation accuracy
        self.val_mae_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        """
        Generic step, used both for training and validation. Executes model, gathers loss and predictions
        Parameters
        ----------
        batch
            the batch which is to be considered

        Returns
        -------
        loss, predictions, targets

        """
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y

    def training_step(self, batch: Any):
        loss, preds, targets = self.step(batch)

        # log train metrics
        mae = self.train_mae(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/mae", mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any):
        loss, preds, targets = self.step(batch)

        # log val metrics
        mae = self.val_mae(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/mae", mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        mae = self.val_mae.compute()  # get val accuracy from current epoch
        self.val_mae_best.update(mae)
        self.log(
            "val/mae_best", self.val_mae_best.compute(), on_epoch=True, prog_bar=True
        )

    def test_step(self, batch: Any):
        loss, preds, targets = self.step(batch)

        # log test metrics
        mae = self.test_mae(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/mae", mae, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_mae.reset()
        self.test_mae.reset()
        self.val_mae.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
        )
        return optimizer