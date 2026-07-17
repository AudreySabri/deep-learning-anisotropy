import logging
import os
import resource
import psutil
from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

def fullname(cls: Any) -> str:
    """Function to return the full name of a particular class, used for hydra instantiate _target_"""
    module = cls.__module__
    if (
        module is None or module == str.__class__.__module__
    ):  # don't want to return 'builtins'
        return cls.__name__
    return module + "." + cls.__name__

def debug_function(x: float):
    """Debugging function"""
    return x**2

def _get_memory_usage_mb():
    """Return current process memory usage in MB."""
    if psutil is not None:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 ** 2
    # fallback to resource; ru_maxrss is kilobytes on Linux
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_maxrss / 1024

def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""
    logger = logging.getLogger(name)
    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))


    return logger

log = get_logger(__name__)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    trainer: pl.Trainer,
) -> None:
    # pylint: disable = protected-access
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
    - number of model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        for _, callback in config["callbacks"].items():
            hparams[f"callbacks/{str(callback._target_)}"] = callback

    # send hparams to all loggers
    trainer.logger.log_hyperparams(
        OmegaConf.to_container(
            hparams,
            resolve=False,
            throw_on_missing=False,
            )
        )
