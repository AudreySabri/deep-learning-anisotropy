import logging
import os
import resource
import psutil, shutil
from typing import Any

import torch
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

def save_checkpoint(state, is_best, save_dir):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        save_dir: (string) folder where parameters are to be saved
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(save_dir, 'last.pth.tar')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_dir, 'best.pth.tar'))

