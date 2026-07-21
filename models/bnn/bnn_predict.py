import logging
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

log = logging.getLogger(__name__)

def BNNPredictor(predictive, dataloader, device, scaler=None):
    """
    Predict using a Pyro model and guide.

    Returns:
        dict: Dictionary containing the predictions and their statistics.
        pd.DataFrame: DataFrame containing the predictions and their statistics.
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
    """
    Convert prediction summary to a dictionary.

    Returns:
        dict: Dictionary containing the predictions and their statistics.
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