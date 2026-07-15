import os
import re
import logging
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def plot_bnn_predictions(results_dict, target_features, save_dir, filename="predictions.png"):
    """Plots the predictions and saves the plot to model_dir
    
    Args:
        results_dict: (dict) Dictionary containing the predictions and their statistics.
        target_features: (list) List of target feature names.
        save_dir: (str) Directory to save the plot.
        filename: (str) Name of the file to save the plot.
    """

    mpl.rcParams['agg.path.chunksize'] = 1000
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Extract the mean predictions and ground truth from the results_dict
    pred_keys = sorted([k for k in results_dict.keys() if k.startswith("mean_")])
    ground_truth_keys = sorted([k for k in results_dict.keys() if k.startswith("ground_truth_")])
    lower_keys = sorted([k for k in results_dict.keys() if k.startswith("5%_")])
    upper_keys = sorted([k for k in results_dict.keys() if k.startswith("95%_")])
    
    predictions = np.column_stack([results_dict[key] for key in pred_keys]) if pred_keys else np.array([])
    ground_truth = np.column_stack([results_dict[key] for key in ground_truth_keys]) if ground_truth_keys else np.array([])
    lower_bound = np.column_stack([results_dict[key] for key in lower_keys]) if lower_keys else np.array([])
    upper_bound = np.column_stack([results_dict[key] for key in upper_keys]) if upper_keys else np.array([])
    n_features = len(target_features)

     # Calculate metrics
    mae = np.mean(np.abs(predictions - ground_truth), axis=0)
    mse = np.mean((predictions - ground_truth)**2, axis=0)
    r2 = 1 - np.sum((ground_truth - predictions)**2, axis=0) / np.sum((ground_truth - np.mean(ground_truth, axis=0))**2, axis=0)

    # Create subplots
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 7*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx in range(n_features):
        ax = axes[idx]

        # Extract data for this feature and plot
        pred = predictions[:, idx]
        true = ground_truth[:, idx]
        lower_bound_feature = lower_bound[:, idx]
        upper_bound_feature = upper_bound[:, idx]
        scatter = ax.scatter(true, 
                             pred, 
                             alpha=0.5,
                             label='Predictions')

        # Plot the uncertainty bounds
        sort_idx = np.argsort(true)
        true_sorted = true[sort_idx]
        lower_sorted = lower_bound_feature[sort_idx]
        upper_sorted = upper_bound_feature[sort_idx]
        fill = ax.fill_between(
            true_sorted,
            lower_sorted,
            upper_sorted,
            color='green',
            alpha=0.1,
            label='Uncertainty interval'
            )
        
        # Add perfect prediction line
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, 
                linewidth=2, label='Perfect prediction')
        
        # Add metrics to plot
        ax.text(0.05, 0.95, f'MAE: {mae[idx]:.3f}\nR²: {r2[idx]:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize='xx-large')

        # Labels and title
        ax.set_xlabel('Ground Truth', fontsize='x-large')
        ax.set_ylabel('Predictions', fontsize='x-large')
        ax.set_title(f'{target_features[idx]}', fontsize='xx-large')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=15)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    # Save the plot to the specified directory
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

    logging.info(f"Saved prediction plots to {save_dir}")
