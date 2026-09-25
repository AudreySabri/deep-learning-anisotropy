import logging
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

log = logging.getLogger(__name__)

def BNNPredictor(predictive, dataloader, device, scaler=None):
    """Run prediction with a Pyro `Predictive` object over a dataloader.

    Args:
        predictive (Callable): A Pyro `Predictive` that returns
            posterior predictive samples when called as `predictive(x)`.
        dataloader (Iterable): PyTorch dataloader yielding (inputs, targets).
        device (str or torch.device): Device to run inference on.
        scaler (optional): Optional scaler with `inverse_transform` used to
            convert scaled outputs back to original units.

    Returns:
        tuple: `(pred_dict, pred_df)` where `pred_dict` maps metric names to
            numpy arrays and `pred_df` is a pandas DataFrame of the same data.
    """
    pred_summary = {}

    for x_test, y_test in dataloader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        samples = predictive(x_test, y=None)
        if scaler is not None:
            for i in range(len(samples["obs"])): 
                samples["obs"][i] = torch.tensor(
                scaler.inverse_transform(samples["obs"][i].cpu()),
                dtype=samples["obs"][i].dtype
                )
            y_test = torch.tensor(
            scaler.inverse_transform(y_test.cpu()),
            dtype=y_test.dtype
            )
        batch_summary = summary(samples, ground_truth=y_test)
        for param_name, metrics in batch_summary.items():
            if param_name not in pred_summary:
                pred_summary[param_name] = {}
            for metric_name, metric_value in metrics.items():
                pred_summary[param_name].setdefault(metric_name, []).append(metric_value)
    logging.info("Prediction completed. Mean MAE per sample=%05.3f", np.mean(pred_summary["obs"]["mae"]))
    pred_dict = results_to_dict(pred_summary)
    pred_df = pd.DataFrame(pred_dict)
    return pred_dict, pred_df
    

def summary(samples, ground_truth = None):
    """Compute per-site summary statistics from posterior samples.

    Calculates mean, standard deviation, and empirical 5/95 percentiles for
    each sampled site. When `ground_truth` is provided for the observation
    site (`"obs"`), also compute MAE and attach the ground truth array.

    Args:
        samples (dict): Mapping from site name to tensor of samples.
        ground_truth (Tensor, optional): True targets for MAE computation.

    Returns:
        dict: Nested mapping site -> statistic name -> tensor/value.
    """
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        }
        if k == "obs":
            if ground_truth is not None:
                site_stats[k]["mae"] = mean_absolute_error(ground_truth.cpu(), torch.mean(v, 0).cpu())
                site_stats[k]["ground_truth"] = ground_truth
    
    return site_stats

def results_to_dict(prediction_summary):
    """Flatten the prediction summary for saving or plotting.

    Converts the nested summary for the observation site into a dict mapping
    column names (e.g. 'mean_0', '5%_1', 'ground_truth_2') to 1D numpy arrays.

    Args:
        prediction_summary (dict): Output from `summary()` for multiple batches
            concatenated per-batch into lists.

    Returns:
        dict: Flattened mapping of metric/feature to numpy array.
    """
    y = prediction_summary["obs"]
    results_dict = {}
    for metric_name, metric_value in y.items():
        if metric_value is not None:
            if isinstance(metric_value[0], torch.Tensor):
                metric_tensor = torch.cat([t.detach() if torch.is_tensor(t) else torch.tensor(t) 
                    for t in metric_value], dim=0).cpu().numpy()
                for i in range(metric_tensor.shape[1]):
                    results_dict[f"{metric_name}_{i}"] = metric_tensor[:, i]
    return results_dict