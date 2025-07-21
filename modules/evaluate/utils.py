import os
from typing import List, Union
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
from matplotlib.patches import Wedge
from numpy import bool_
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from modules.training.ts_modeling import (process_predictions,
                                          evaluate_pcc,
                                          evaluate_mae,
                                          filter_ds,
                                          process_sep_events,
                                          plot_error_hist)


def evaluate_plot(
        model: tf.keras.Model,
        X_test: Union[np.ndarray, List[np.ndarray]],
        y_test: np.ndarray,
        lower_threshold: float = None,
        upper_threshold: float = None,
        N_samples: int = 1000,
        seed: int = 42,
        title: str = "",
        model_type: str = 'features_reg',
        test_directory: str = '',
        split: str = 'test',
        inputs_to_use: List[str] = None,
        outputs_to_use: List[str] = None,
        add_slope: bool = False,
        use_cme: bool = True,
        cme_speed_threshold: float = 0,
        show_avsp: bool = True,
        current_time: str = None,
        wandb_cls: Optional[wandb.sdk.wandb_run.Run] = None,

):
    """
    Evaluates a given model using with plots

    Args:
        model: The trained model to evaluate.
        X_test:  Test features.
        y_test:  True target values for the test set.
        lower_threshold: The lower bound threshold for y_test to be included in the evaluation.
        upper_threshold: The upper bound threshold for y_test to be included in the evaluation.
        N_samples: Number of samples to use for the evaluation.
        seed: Random seed for the evaluation.
        title: Title for the plots.
        model_type: The type of model to use (features, features_reg, features_dec, features_reg_dec).
        test_directory: The directory containing SEP event files to process.
        split: The split being evaluated (e.g., 'test', 'validation').
        inputs_to_use: The input features to use for the evaluation.
        outputs_to_use: The output features to use for the evaluation.
        add_slope: Whether to add the slope to the model.
        use_cme: Whether to use CME data in the evaluation.
        cme_speed_threshold: The threshold for CME speed.
        show_avsp: Whether to show the actual vs predicted plot.
        current_time: The current time to append to the file names.
        wandb_cls: The wandb class instance to log the evaluation results

    Returns:
        None

    """
    X_test_filtered, y_test_filtered = filter_ds(
        X_test, y_test,
        low_threshold=lower_threshold,
        high_threshold=upper_threshold,
        N=N_samples, seed=seed)

    # Process SEP event files in the specified directory
    filenames = process_sep_events(
        test_directory,
        model,
        title=title,
        inputs_to_use=inputs_to_use,
        add_slope=add_slope,
        outputs_to_use=outputs_to_use,
        show_avsp=show_avsp,
        using_cme=use_cme,
        cme_speed_threshold=cme_speed_threshold)

    # Log the plot to wandb
    for filename in filenames:
        log_title = os.path.basename(filename)
        wandb_cls.log({f'{split}ing_{log_title}': wandb_cls.Image(filename)})

    # Process SEP event files in the specified directory
    file_path = plot_repr_corr_dist(
        model, X_test_filtered, y_test_filtered,
        title + f"_{split}",
        model_type=model_type
    )
    wandb_cls.log({f'representation_correlation_colored_plot_{split}': wandb_cls.Image(file_path)})
    print('file_path: ' + file_path)

    # Log the testing t-SNE plot to wandb
    stage1_file_path = plot_tsne_delta(
        model,
        X_test_filtered, y_test_filtered, title,
        f'stage2_{split}ing',
        model_type=model_type,
        save_tag=current_time, seed=seed)
    wandb_cls.log({f'stage2_tsne_{split}ing_plot': wandb_cls.Image(stage1_file_path)})
    print('stage1_file_path: ' + stage1_file_path)

    # Plot the error histograms on the testing set
    filename = plot_error_hist(
        model,
        X_test, y_test,
        sample_weights=None,
        title=title,
        prefix=f'{split}ing')
    wandb_cls.log({f"{split}ing_error_hist": wandb_cls.Image(filename)})


def evaluate_mae_pcc(
        model: tf.keras.Model,
        X_test: Union[np.ndarray, List[np.ndarray]],
        y_test: np.ndarray,
        above_threshold: float = None,
        below_threshold: float = None,
        use_dict: bool = False,
        wandb_cls: Optional[wandb.sdk.wandb_run.Run] = None,
):
    """
    Evaluates a given model using Mean Absolute Error (MAE) and Pearson Correlation Coefficient (PCC) metrics.

    Args:
        model: The trained model to evaluate.
        X_test:  Test features.
        y_test:  True target values for the test set.
        above_threshold: The upper bound threshold for y_test to be included in the evaluation.
        below_threshold: The lower bound threshold for y_test to be included in the evaluation.
        use_dict: Whether to use a dictionary to store the evaluation results.
        wandb_cls: The wandb class instance to log the evaluation results.

    Returns:
        None

    """

    # evaluate the model error on test set
    error_mae = evaluate_mae(model, X_test, y_test, use_dict=use_dict)
    print(f'mae error: {error_mae}')
    wandb_cls.log({"mae": error_mae})

    # evaluate the model correlation on test set
    error_pcc = evaluate_pcc(model, X_test, y_test, use_dict=use_dict)
    print(f'pcc error: {error_pcc}')
    wandb_cls.log({"pcc": error_pcc})

    # evaluate the model error for rare samples on test set
    error_mae_cond = evaluate_mae(
        model, X_test, y_test,
        above_threshold=above_threshold,
        below_threshold=below_threshold,
        use_dict=use_dict)
    print(f'mae error delta >= {above_threshold} test: {error_mae_cond}')
    wandb_cls.log({"mae+": error_mae_cond})

    # evaluate the model correlation for rare samples on test set
    error_pcc_cond = evaluate_pcc(
        model, X_test, y_test,
        above_threshold=above_threshold,
        below_threshold=below_threshold,
        use_dict=use_dict)
    print(f'pcc error delta >= {above_threshold} test: {error_pcc_cond}')
    wandb_cls.log({"pcc+": error_pcc_cond})


def print_statistics(statistics: dict) -> None:
    """
    Prints the statistics for each metric and batch size in a formatted table with values rounded to three decimal places.

    :param statistics: A dictionary containing the statistics for each batch size.
    """
    # Find all unique metrics across batch sizes
    all_metrics = set()
    for stats in statistics.values():
        all_metrics.update(stats.keys())

    # Print header
    header = "Batch Size".ljust(15) + "".join(metric.ljust(20) for metric in sorted(all_metrics))
    print(header)
    print("-" * len(header))

    # Print stats for each batch size
    for batch_size, metrics in sorted(statistics.items()):
        stats_str = f"{str(batch_size).ljust(15)}"
        for metric in sorted(all_metrics):
            mean = f"{metrics[metric]['mean']:.3f}" if metric in metrics else 'N/A'
            std = f"{metrics[metric]['std']:.3f}" if metric in metrics else 'N/A'
            stats_str += f"{f'{mean} (±{std})'.ljust(20)}"
        print(stats_str)


def update_tracking(results_tracking: dict, batch_size: int, metrics: dict) -> None:
    """
    Update the results tracking dictionary with the new metrics from a model run.

    :param results_tracking: A dictionary to track the metrics for each method (batch size).
    :param batch_size: The batch size used in the model run.
    :param metrics: A dictionary containing the metrics from the model run.
    """
    if batch_size not in results_tracking:
        results_tracking[batch_size] = []

    # Append the metrics to the list for this batch size
    results_tracking[batch_size].append(metrics)


# Define the function to calculate statistics
def calculate_statistics(results_tracking: dict) -> dict:
    """
    Calculate the mean and standard deviation for each metric across all runs.

    :param results_tracking: A dictionary with batch size keys and lists of metric dictionaries as values.
    :return: A dictionary with the calculated statistics for each metric.
    """
    statistics = {}
    for batch_size, runs in results_tracking.items():
        # Initialize a dictionary for this batch size
        stats_for_batch = {}
        # Exclude non-numeric metrics like 'plot'
        numeric_metrics = {metric for metric in runs[0] if isinstance(runs[0][metric], (int, float))}

        # Calculate mean and standard deviation for each numeric metric
        for metric in numeric_metrics:
            values = [run[metric] for run in runs]
            stats_for_batch[metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }

        # Assign the stats to the corresponding batch size
        statistics[batch_size] = stats_for_batch

    return statistics


def split_combined_joint_weights_indices(
        combined_weights: np.ndarray,
        combined_indices: List[Tuple[int, int]],
        len_train: int, len_val: int) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray, List[Tuple[int, int]]]:
    """
    Splits the combined joint weights and indices back into the original training and validation joint weights and indices.

    Parameters:
        combined_weights (np.ndarray): The combined joint weights of training and validation sets.
        combined_indices (List[Tuple[int, int]]): The combined index pairs mapping to ya and yb in the original dataset.
        len_train (int): The length of the original training set.
        len_val (int): The length of the original validation set.

    Returns:
        Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray, List[Tuple[int, int]]]:
        Tuple containing the training and validation joint weights and index pairs.
    """
    train_weights, train_indices = [], []
    val_weights, val_indices = [], []

    for weight, (i, j) in zip(combined_weights, combined_indices):
        if i < len_train and j < len_train:
            train_weights.append(weight)
            train_indices.append((i, j))
        elif i >= len_train and j >= len_train:
            val_weights.append(weight)
            val_indices.append((i - len_train, j - len_train))

    return np.array(train_weights), train_indices, np.array(val_weights), val_indices


def plot_repr_corr_density(model, X, y, title, model_type='features'):
    """
    Plots a heatmap showing the concentration of points in the representation space and target value space.

    Parameters:
    - model: The trained model used to transform input features into a representation space.
    - X: Input features (NumPy array or compatible).
    - y: Target values (NumPy array or compatible).
    - title: Title for the plot.

    Returns:
    - str: File path of the saved plot.
    """
    # Predict representations
    print('In plot_repr_correlation_density')
    if model_type in ['features_reg_dec', 'features_reg', 'features_dec']:
        representations = model.predict(X)[0]  # Assuming the first output is always features
    else:
        representations = model.predict(X)

    # Calculate distances
    distances_target = pdist(y.reshape(-1, 1), 'euclidean')
    distances_repr = pdist(representations, 'euclidean')

    # Normalize distances
    scaler = MinMaxScaler()
    distances_target_norm = scaler.fit_transform(distances_target.reshape(-1, 1)).flatten()
    distances_repr_norm = scaler.fit_transform(distances_repr.reshape(-1, 1)).flatten()

    # Define bins for normalized distances
    bin_width = 0.05
    bins = np.arange(0, 1 + bin_width, bin_width)

    # Create a 2D histogram of normalized distances
    counts, xedges, yedges = np.histogram2d(distances_target_norm, distances_repr_norm, bins=[bins, bins])

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    heatmap = plt.pcolormesh(xedges, yedges, counts.T, cmap='Greys')
    plt.colorbar(heatmap, label='Frequency')

    # Adding frequency labels to each cell
    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            # if counts[i][j] > 0:  # Only add labels to non-zero cells
            plt.text(xedges[i] + bin_width / 2, yedges[j] + bin_width / 2, f'{int(counts[i][j])}',
                     color='tab:blue', ha='center', va='center', fontsize=8)
    #
    plt.xlabel('Normalized Distance in Target Space')
    plt.ylabel('Normalized Distance in Representation Space')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)

    plot_filename = f"{title.replace(' ', '_').lower()}_density_plot.png"
    plt.savefig(plot_filename)
    plt.close()

    return plot_filename


def evaluate_pcc_repr(
        model: tf.keras.Model,
        X_test: Union[np.ndarray, List[np.ndarray]],
        y_test: np.ndarray,
        i_below_threshold: float = None,
        i_above_threshold: float = None,
        j_below_threshold: float = None,
        j_above_threshold: float = None,
        model_type: str = 'features') -> float:
    """
    Evaluates a given model using Pearson Correlation Coefficient (PCC) between
    distances in target values and distances in the representation space.

    Parameters:
    - model (tf.keras.Model): The trained model to evaluate.
    - X_test (np.ndarray or List[np.ndarray]): Test features.
    - y_test (np.ndarray): True target values for the test set.
    - i_below_threshold (float, optional): The lower bound threshold for y_test[i] to be included in PCC calculation.
    - i_above_threshold (float, optional): The upper bound threshold for y_test[i] to be included in PCC calculation.
    - j_below_threshold (float, optional): The lower bound threshold for y_test[j] to be included in PCC calculation.
    - j_above_threshold (float, optional): The upper bound threshold for y_test[j] to be included in PCC calculation.
    - model_type (str): The type of model to use (features, features_reg, features_dec, features_reg_dec).

    Returns:
    - float: The PCC between distances in target values and distances in the representation space.
    """
    # Get representations
    if model_type in ['features_reg_dec', 'features_reg', 'features_dec']:
        representations = model.predict(X_test)[0]  # Assuming the first output is always features
    else:
        representations = model.predict(X_test)

    # Calculate pairwise distances
    distances_target = pdist(y_test.reshape(-1, 1), 'euclidean')
    distances_repr = pdist(representations, 'euclidean')

    # Create mask for filtering based on thresholds
    n = len(y_test)
    mask = np.ones(len(distances_target), dtype=bool)

    if any([i_below_threshold, i_above_threshold, j_below_threshold, j_above_threshold]):
        upper_tri_indices = np.triu_indices(n, k=1)
        for k, (i, j) in enumerate(zip(*upper_tri_indices)):
            i_condition = True
            j_condition = True

            if i_below_threshold is not None:
                i_condition &= y_test[i] <= i_below_threshold
            if i_above_threshold is not None:
                i_condition &= y_test[i] >= i_above_threshold
            if j_below_threshold is not None:
                j_condition &= y_test[j] <= j_below_threshold
            if j_above_threshold is not None:
                j_condition &= y_test[j] >= j_above_threshold

            mask[k] = i_condition and j_condition

    # Apply mask
    filtered_distances_target = distances_target[mask]
    filtered_distances_repr = distances_repr[mask]

    # Normalize distances
    scaler = MinMaxScaler()
    distances_target_norm = scaler.fit_transform(filtered_distances_target.reshape(-1, 1)).flatten()
    distances_repr_norm = scaler.fit_transform(filtered_distances_repr.reshape(-1, 1)).flatten()

    # Calculate Pearson correlation
    r, _ = pearsonr(distances_target_norm, distances_repr_norm)

    return r


def plot_repr_corr_dist(model, X, y, title, model_type='features'):
    """
    Plots the correlation between distances in target values and distances in the representation space,
    with each point colored based on the pair of labels.

    Parameters:
    - model: The trained model used to transform input features into a representation space.
    - X: Input features (NumPy array or compatible).
    - y: Target values and labels (NumPy array or compatible).
    - title: Title for the plot.
    - model_type: The type of model to use (features, features_reg, features_dec, features_reg_dec).

    Returns:
    - Plots the representation distance correlation plot with label-based coloring.
    """
    print('In plot_repr_corr_dist')
    if model_type in ['features_reg_dec', 'features_reg', 'features_dec', 'features_cls']:
        representations = model.predict(X)[0]  # Assuming the first output is always features
    elif model_type == 'dict':
        representations = model.predict(X)['repr']
    else:
        representations = model.predict(X)

    print('Calculating the pairwise distances')
    distances_target = pdist(y.reshape(-1, 1), 'euclidean')
    distances_repr = pdist(representations, 'euclidean')

    scaler = MinMaxScaler()
    distances_target_norm = scaler.fit_transform(distances_target.reshape(-1, 1)).flatten()
    distances_repr_norm = scaler.fit_transform(distances_repr.reshape(-1, 1)).flatten()

    print('Calculating the spearman rank correlation')
    r, _ = pearsonr(distances_target_norm, distances_repr_norm)

    print('Assigning colors based on labels')

    def get_color(label):
        if label < -0.5:
            return 'blue'
        elif label > 0.5:
            return 'red'
        else:
            return 'gray'

    label_pairs = [(y[i], y[j]) for i in range(len(y)) for j in range(i + 1, len(y))]
    colors = [(get_color(label1), get_color(label2)) for label1, label2 in label_pairs]

    # Function to draw a half-colored dot
    def draw_half_colored_dot(ax, x, y, color1, color2, size=0.01):
        wedge1 = Wedge((x, y), size, 0, 180, color=color1)
        wedge2 = Wedge((x, y), size, 180, 360, color=color2)
        ax.add_patch(wedge1)
        ax.add_patch(wedge2)

    print('Plotting with label-based colors')
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (x, y) in enumerate(zip(distances_target_norm, distances_repr_norm)):
        color1, color2 = colors[i]
        draw_half_colored_dot(ax, x, y, color1, color2, size=0.005)  # Adjust size as needed

    ax.plot([0, 1], [0, 1], 'k--')  # Perfect fit diagonal
    ax.set_xlabel('Normalized Distance in Target Space')
    ax.set_ylabel('Normalized Distance in Representation Space')
    ax.set_title(f'{title}\nRepresentation Space Correlation (pearson r= {r:.2f})')
    ax.grid(True)
    plt.savefig(f"representation_correlation_labels_{title}.png")
    plt.close()

    return f"representation_correlation_labels_{title}.png"

def plot_sarcos_corr(model, X, y, title, model_type='features_reg', lower_threshold=None, upper_threshold=None):
    """
    Plots the correlation between distances in target values and distances in the representation space,
    with each point colored based on the target value categories:
    - Blue for values below lower_threshold
    - Red for values above upper_threshold  
    - Gray for values between thresholds

    Parameters:
    - model: The trained model used to transform input features into a representation space.
    - X: Input features (NumPy array or compatible).
    - y: Target values (NumPy array or compatible).
    - title: Title for the plot.
    - model_type: The type of model to use (features, features_reg, features_dec, features_reg_dec).
    - lower_threshold: Lower bound threshold for rare values.
    - upper_threshold: Upper bound threshold for rare values.

    Returns:
    - Path to the saved correlation plot.
    """
    print('In plot_sarcos_corr')
    if model_type in ['features_reg_dec', 'features_reg', 'features_dec', 'features_cls']:
        representations = model.predict(X)[0]  # Assuming the first output is always features
    elif model_type == 'dict':
        representations = model.predict(X)['repr']
    else:
        representations = model.predict(X)

    print('Calculating the pairwise distances')
    distances_target = pdist(y.reshape(-1, 1), 'euclidean')
    distances_repr = pdist(representations, 'euclidean')

    scaler = MinMaxScaler()
    distances_target_norm = scaler.fit_transform(distances_target.reshape(-1, 1)).flatten()
    distances_repr_norm = scaler.fit_transform(distances_repr.reshape(-1, 1)).flatten()

    print('Calculating the pearson correlation')
    r, _ = pearsonr(distances_target_norm, distances_repr_norm)

    print('Assigning colors based on value ranges')

    def get_color(label):
        if lower_threshold is not None and label < lower_threshold:
            return 'blue'  # Values below lower threshold
        elif upper_threshold is not None and label > upper_threshold:
            return 'red'   # Values above upper threshold
        else:
            return 'gray'  # Values in the middle range

    label_pairs = [(y[i], y[j]) for i in range(len(y)) for j in range(i + 1, len(y))]
    colors = [(get_color(label1), get_color(label2)) for label1, label2 in label_pairs]

    # Function to draw a half-colored dot
    def draw_half_colored_dot(ax, x, y, color1, color2, size=0.01):
        wedge1 = Wedge((x, y), size, 0, 180, color=color1)
        wedge2 = Wedge((x, y), size, 180, 360, color=color2)
        ax.add_patch(wedge1)
        ax.add_patch(wedge2)

    print('Plotting with three-color scheme')
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (x, y) in enumerate(zip(distances_target_norm, distances_repr_norm)):
        color1, color2 = colors[i]
        draw_half_colored_dot(ax, x, y, color1, color2, size=0.005)  # Adjust size as needed

    ax.plot([0, 1], [0, 1], 'k--')  # Perfect fit diagonal
    ax.set_xlabel('Normalized Distance in Target Space')
    ax.set_ylabel('Normalized Distance in Representation Space')
    
    # Add description of color coding to title
    title_with_thresholds = f'{title}\nRepresentation Space Correlation (pearson r= {r:.2f})\n'
    title_with_thresholds += f'Blue: below {lower_threshold:.2f}, Red: above {upper_threshold:.2f}, Gray: in between'
    
    ax.set_title(title_with_thresholds)
    ax.grid(True)
    
    file_path = f"representation_correlation_sarcos_{title}.png"
    plt.savefig(file_path)
    plt.close()

    return file_path

def plot_onp_corr(
    model, 
    X: np.ndarray, 
    y: np.ndarray, 
    title: str, 
    model_type: str = 'features_reg', 
    lower_threshold: float = None, 
    upper_threshold: float = None
) -> str:
    """
    Plots the correlation between distances in target values and distances in the representation space
    for Online News Popularity (ONP) dataset, with each point colored based on the target value categories:
    - Blue for values below lower_threshold
    - Red for values above upper_threshold  
    - Gray for values between thresholds

    Parameters:
    - model: The trained model used to transform input features into a representation space.
    - X: Input features (NumPy array).
    - y: Target values (NumPy array).
    - title: Title for the plot.
    - model_type: The type of model to use (features, features_reg, features_dec, features_reg_dec).
    - lower_threshold: Lower bound threshold for rare values.
    - upper_threshold: Upper bound threshold for rare values.

    Returns:
    - Path to the saved correlation plot.
    """
    print('In plot_onp_corr')
    if model_type in ['features_reg_dec', 'features_reg', 'features_dec', 'features_cls']:
        representations = model.predict(X)[0]  # Assuming the first output is always features
    elif model_type == 'dict':
        representations = model.predict(X)['repr']
    else:
        representations = model.predict(X)

    print('Calculating the pairwise distances')
    distances_target = pdist(y.reshape(-1, 1), 'euclidean')
    distances_repr = pdist(representations, 'euclidean')

    scaler = MinMaxScaler()
    distances_target_norm = scaler.fit_transform(distances_target.reshape(-1, 1)).flatten()
    distances_repr_norm = scaler.fit_transform(distances_repr.reshape(-1, 1)).flatten()

    print('Calculating the pearson correlation')
    r, _ = pearsonr(distances_target_norm, distances_repr_norm)

    print('Assigning colors based on value ranges')

    def get_color(label):
        if lower_threshold is not None and label < lower_threshold:
            return 'blue'  # Values below lower threshold
        elif upper_threshold is not None and label > upper_threshold:
            return 'red'   # Values above upper threshold
        else:
            return 'gray'  # Values in the middle range

    label_pairs = [(y[i], y[j]) for i in range(len(y)) for j in range(i + 1, len(y))]
    colors = [(get_color(label1), get_color(label2)) for label1, label2 in label_pairs]

    # Function to draw a half-colored dot
    def draw_half_colored_dot(ax, x, y, color1, color2, size=0.01):
        wedge1 = Wedge((x, y), size, 0, 180, color=color1)
        wedge2 = Wedge((x, y), size, 180, 360, color=color2)
        ax.add_patch(wedge1)
        ax.add_patch(wedge2)

    print('Plotting with three-color scheme')
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (x, y) in enumerate(zip(distances_target_norm, distances_repr_norm)):
        color1, color2 = colors[i]
        draw_half_colored_dot(ax, x, y, color1, color2, size=0.005)  # Adjust size as needed

    ax.plot([0, 1], [0, 1], 'k--')  # Perfect fit diagonal
    ax.set_xlabel('Normalized Distance in Target Space')
    ax.set_ylabel('Normalized Distance in Representation Space')
    
    # Add description of color coding to title
    title_with_thresholds = f'{title}\nRepresentation Space Correlation (pearson r= {r:.2f})\n'
    title_with_thresholds += f'Blue: below {lower_threshold:.2f}, Red: above {upper_threshold:.2f}, Gray: in between'
    
    ax.set_title(title_with_thresholds)
    ax.grid(True)
    
    file_path = f"representation_correlation_onp_{title}.png"
    plt.savefig(file_path)
    plt.close()

    return file_path

def plot_asc_corr(
    model, 
    X: np.ndarray, 
    y: np.ndarray, 
    title: str, 
    model_type: str = 'features_reg', 
    lower_threshold: float = None, 
    upper_threshold: float = None
) -> str:
    """
    Plots the correlation between distances in target values and distances in the representation space
    for All State Claims (ASC) dataset, with each point colored based on the target value categories:
    - Blue for values below lower_threshold
    - Red for values above upper_threshold  
    - Gray for values between thresholds

    Parameters:
    - model: The trained model used to transform input features into a representation space.
    - X: Input features (NumPy array).
    - y: Target values (NumPy array).
    - title: Title for the plot.
    - model_type: The type of model to use (features, features_reg, features_dec, features_reg_dec).
    - lower_threshold: Lower bound threshold for rare values.
    - upper_threshold: Upper bound threshold for rare values.

    Returns:
    - Path to the saved correlation plot.
    """
    print('In plot_asc_corr')
    if model_type in ['features_reg_dec', 'features_reg', 'features_dec', 'features_cls']:
        representations = model.predict(X)[0]  # Assuming the first output is always features
    elif model_type == 'dict':
        representations = model.predict(X)['repr']
    else:
        representations = model.predict(X)

    print('Calculating the pairwise distances')
    distances_target = pdist(y.reshape(-1, 1), 'euclidean')
    distances_repr = pdist(representations, 'euclidean')

    scaler = MinMaxScaler()
    distances_target_norm = scaler.fit_transform(distances_target.reshape(-1, 1)).flatten()
    distances_repr_norm = scaler.fit_transform(distances_repr.reshape(-1, 1)).flatten()

    print('Calculating the pearson correlation')
    r, _ = pearsonr(distances_target_norm, distances_repr_norm)

    print('Assigning colors based on value ranges')

    def get_color(label):
        if lower_threshold is not None and label < lower_threshold:
            return 'blue'  # Values below lower threshold
        elif upper_threshold is not None and label > upper_threshold:
            return 'red'   # Values above upper threshold
        else:
            return 'gray'  # Values in the middle range

    label_pairs = [(y[i], y[j]) for i in range(len(y)) for j in range(i + 1, len(y))]
    colors = [(get_color(label1), get_color(label2)) for label1, label2 in label_pairs]

    # Function to draw a half-colored dot
    def draw_half_colored_dot(ax, x, y, color1, color2, size=0.01):
        wedge1 = Wedge((x, y), size, 0, 180, color=color1)
        wedge2 = Wedge((x, y), size, 180, 360, color=color2)
        ax.add_patch(wedge1)
        ax.add_patch(wedge2)

    print('Plotting with three-color scheme')
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (x, y) in enumerate(zip(distances_target_norm, distances_repr_norm)):
        color1, color2 = colors[i]
        draw_half_colored_dot(ax, x, y, color1, color2, size=0.005)  # Adjust size as needed

    ax.plot([0, 1], [0, 1], 'k--')  # Perfect fit diagonal
    ax.set_xlabel('Normalized Distance in Target Space')
    ax.set_ylabel('Normalized Distance in Representation Space')
    
    # Add description of color coding to title
    title_with_thresholds = f'{title}\nRepresentation Space Correlation (pearson r= {r:.2f})\n'
    title_with_thresholds += f'Blue: below {lower_threshold:.2f}, Red: above {upper_threshold:.2f}, Gray: in between'
    
    ax.set_title(title_with_thresholds)
    ax.grid(True)
    
    file_path = f"representation_correlation_asc_{title}.png"
    plt.savefig(file_path)
    plt.close()

    return file_path


def plot_blogf_corr(
    model, 
    X: np.ndarray, 
    y: np.ndarray, 
    title: str, 
    model_type: str = 'features_reg', 
    freq_threshold: float = None, 
    rare_threshold: float = None
) -> str:
    """
    Plots the correlation between distances in target values and distances in the representation space
    for Blog Feedback dataset, with each point colored based on the target value categories:
    - Blue for frequent values (y < log10(4))
    - Green for medium values (log10(4) < y < log10(40))
    - Red for rare values (y > log10(40))

    Parameters:
    - model: The trained model used to transform input features into a representation space.
    - X: Input features (NumPy array).
    - y: Target values (log10 of comments in next 24h) (NumPy array).
    - title: Title for the plot.
    - model_type: The type of model to use (features, features_reg, features_dec, features_reg_dec).
    - freq_threshold: Threshold for frequent values (default: log10(4)).
    - rare_threshold: Threshold for rare values (default: log10(40)).

    Returns:
    - Path to the saved correlation plot.
    """
    print('In plot_blogf_corr')
    # Set default thresholds if not provided
    if freq_threshold is None:
        freq_threshold = np.log10(4)
    if rare_threshold is None:
        rare_threshold = np.log10(40)
        
    if model_type in ['features_reg_dec', 'features_reg', 'features_dec', 'features_cls']:
        representations = model.predict(X)[0]  # Assuming the first output is always features
    elif model_type == 'dict':
        representations = model.predict(X)['repr']
    else:
        representations = model.predict(X)

    print('Calculating the pairwise distances')
    distances_target = pdist(y.reshape(-1, 1), 'euclidean')
    distances_repr = pdist(representations, 'euclidean')

    scaler = MinMaxScaler()
    distances_target_norm = scaler.fit_transform(distances_target.reshape(-1, 1)).flatten()
    distances_repr_norm = scaler.fit_transform(distances_repr.reshape(-1, 1)).flatten()

    print('Calculating the pearson correlation')
    r, _ = pearsonr(distances_target_norm, distances_repr_norm)

    print('Assigning colors based on value ranges')

    def get_color(label):
        if label < freq_threshold:
            return 'blue'  # Frequent values: y < log10(4)
        elif label > rare_threshold:
            return 'red'   # Rare values: y > log10(40)
        else:
            return 'green'  # Medium values: log10(4) < y < log10(40)

    label_pairs = [(y[i], y[j]) for i in range(len(y)) for j in range(i + 1, len(y))]
    colors = [(get_color(label1), get_color(label2)) for label1, label2 in label_pairs]

    # Function to draw a half-colored dot
    def draw_half_colored_dot(ax, x, y, color1, color2, size=0.01):
        wedge1 = Wedge((x, y), size, 0, 180, color=color1)
        wedge2 = Wedge((x, y), size, 180, 360, color=color2)
        ax.add_patch(wedge1)
        ax.add_patch(wedge2)

    print('Plotting with three-color scheme')
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (x, y) in enumerate(zip(distances_target_norm, distances_repr_norm)):
        color1, color2 = colors[i]
        draw_half_colored_dot(ax, x, y, color1, color2, size=0.005)  # Adjust size as needed

    ax.plot([0, 1], [0, 1], 'k--')  # Perfect fit diagonal
    ax.set_xlabel('Normalized Distance in Target Space')
    ax.set_ylabel('Normalized Distance in Representation Space')
    
    # Add description of color coding to title
    title_with_thresholds = f'{title}\nRepresentation Space Correlation (pearson r= {r:.2f})\n'
    title_with_thresholds += f'Blue: frequent (y < {freq_threshold:.2f}), Green: medium, Red: rare (y > {rare_threshold:.2f})'
    
    ax.set_title(title_with_thresholds)
    ax.grid(True)
    
    file_path = f"representation_correlation_blogf_{title}.png"
    plt.savefig(file_path)
    plt.close()

    return file_path


def plot_sep_corr(model, X, y, title, model_type='features', sep_threshold=2.30258509299):
    """
    Plots the correlation between distances in target values and distances in the representation space,
    with each point colored based on the pair of labels using a SEP threshold.

    Parameters:
    - model: The trained model used to transform input features into a representation space.
    - X: Input features (NumPy array or compatible).
    - y: Target values and labels (NumPy array or compatible).
    - title: Title for the plot.
    - model_type: The type of model to use (features, features_reg, features_dec, features_reg_dec).
    - sep_threshold: Threshold value to distinguish SEP events from non-SEP events.

    Returns:
    - Plots the representation distance correlation plot with label-based coloring.
    """
    print('In plot_sep_corr')
    if model_type in ['features_reg_dec', 'features_reg', 'features_dec', 'features_cls']:
        representations = model.predict(X)[0]  # Assuming the first output is always features
    elif model_type == 'dict':
        representations = model.predict(X)['repr']
    else:
        representations = model.predict(X)

    print('Calculating the pairwise distances')
    distances_target = pdist(y.reshape(-1, 1), 'euclidean')
    distances_repr = pdist(representations, 'euclidean')

    scaler = MinMaxScaler()
    distances_target_norm = scaler.fit_transform(distances_target.reshape(-1, 1)).flatten()
    distances_repr_norm = scaler.fit_transform(distances_repr.reshape(-1, 1)).flatten()

    print('Calculating the spearman rank correlation')
    r, _ = pearsonr(distances_target_norm, distances_repr_norm)

    print('Assigning colors based on labels')

    def get_color(label):
        if label >= sep_threshold:
            return 'red'
        else:
            return 'gray'

    label_pairs = [(y[i], y[j]) for i in range(len(y)) for j in range(i + 1, len(y))]
    colors = [(get_color(label1), get_color(label2)) for label1, label2 in label_pairs]

    # Function to draw a half-colored dot
    def draw_half_colored_dot(ax, x, y, color1, color2, size=0.01):
        wedge1 = Wedge((x, y), size, 0, 180, color=color1)
        wedge2 = Wedge((x, y), size, 180, 360, color=color2)
        ax.add_patch(wedge1)
        ax.add_patch(wedge2)

    print('Plotting with label-based colors')
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (x, y) in enumerate(zip(distances_target_norm, distances_repr_norm)):
        color1, color2 = colors[i]
        draw_half_colored_dot(ax, x, y, color1, color2, size=0.005)  # Adjust size as needed

    ax.plot([0, 1], [0, 1], 'k--')  # Perfect fit diagonal
    ax.set_xlabel('Normalized Distance in Target Space')
    ax.set_ylabel('Normalized Distance in Representation Space')
    ax.set_title(f'{title}\nRepresentation Space Correlation (pearson r= {r:.2f})')
    ax.grid(True)
    plt.savefig(f"representation_correlation_labels_{title}.png")
    plt.close()

    return f"representation_correlation_labels_{title}.png"



def plot_repr_correlation(model, X, y, title, model_type='features'):
    """
    Plots the correlation between distances in target values and distances in the representation space, 
    with point density indicated by color.

    Parameters:
    - model: The trained model used to transform input features into a representation space.
    - X: Input features (NumPy array or compatible).
    - y: Target values (NumPy array or compatible).
    - title: Title for the plot.
    - model_type: The type of model to use (features, features_reg, features_dec, features_reg_dec).

    Returns:
    - Plots the representation distance correlation plot with density coloring.
    """
    print('In plot_repr_correlation')
    if model_type in ['features_reg_dec', 'features_reg', 'features_dec']:
        representations = model.predict(X)[0]  # Assuming the first output is always features
    else:
        representations = model.predict(X)

    print('calculating the pairwise distances')
    distances_target = pdist(y.reshape(-1, 1), 'euclidean')
    distances_repr = pdist(representations, 'euclidean')

    scaler = MinMaxScaler()
    distances_target_norm = scaler.fit_transform(distances_target.reshape(-1, 1)).flatten()
    distances_repr_norm = scaler.fit_transform(distances_repr.reshape(-1, 1)).flatten()

    print('calculating the spearman rank correlation and density')
    r, _ = pearsonr(distances_target_norm, distances_repr_norm)
    xy = np.vstack([distances_target_norm, distances_repr_norm])
    z = gaussian_kde(xy)(xy)  # Calculate density values

    print('plotting with density colors')
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(distances_target_norm, distances_repr_norm, c=z, s=5, edgecolor='none', cmap='viridis',
                          alpha=0.6)
    plt.colorbar(scatter, label='Density')
    plt.plot([0, 1], [0, 1], 'k--')  # Perfect fit diagonal
    plt.xlabel('Normalized Distance in Target Space')
    plt.ylabel('Normalized Distance in Representation Space')
    plt.title(f'{title}\nRepresentation Space Correlation (pearson r= {r:.2f})')
    plt.grid(True)
    plt.savefig(f"representation_correlation_density_{title}.png")
    plt.close()

    return f"representation_correlation_density_{title}.png"


def plot_shepard(features, tsne_result):
    """
    Helper function to create a Shepard plot with normalized distances, including a perfect fit diagonal,
    and displaying the rank correlation in the title.

    Parameters:
    - features: High-dimensional features
    - tsne_result: 2D coordinates from t-SNE

    Returns:
    - Plots the Shepard plot
    """
    # Calculate pairwise distances in the original and reduced space
    print('In plot_shepard')
    distances_original = pdist(features, 'euclidean')
    distances_tsne = pdist(tsne_result, 'euclidean')

    # Normalize distances
    scaler = MinMaxScaler()
    distances_original_norm = scaler.fit_transform(distances_original[:, np.newaxis]).flatten()
    distances_tsne_norm = scaler.fit_transform(distances_tsne[:, np.newaxis]).flatten()

    print('calculating the spearman rank correlation')
    # Calculate Spearman's rank correlation
    r, _ = pearsonr(distances_original_norm, distances_tsne_norm)
    print('plotting now...')
    # Plot normalized distances
    plt.scatter(distances_original_norm, distances_tsne_norm, alpha=0.5, s=1)
    plt.plot([0, 1], [0, 1], 'k--')  # Perfect fit diagonal
    plt.xlabel('Normalized Original Distances')
    plt.ylabel('Normalized t-SNE Distances')
    plt.title(f'Shepard Plot (correlation = {r:.2f})')
    # plt.title(f'Shepard Plot')
    plt.grid(True)

    print('Done with plot_shepard')


def plot_tsne(
        X: np.ndarray,
        y: np.ndarray,
        title: str,
        prefix: str,
        show_plot=False,
        save_tag=None,
        seed=42) -> str:
    """
    Visualizes changes (e.g., in logIntensity) using t-SNE by coloring points based on their values.

    Parameters:
    - X: Input data (NumPy array or compatible).
    - y: Target labels (NumPy array or compatible), representing changes.
    - title: Title for the plot.
    - prefix: Prefix for the file name.
    - show_plot: If True, display the plot in addition to saving it.
    - save_tag: Optional tag to append to the file name.
    - seed: Random seed for t-SNE.

    Returns:
    - The file path of the saved t-SNE plot.
    """

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=seed)
    tsne_result = tsne.fit_transform(X)

    # Plot setup
    fig, axs = plt.subplots(2, 1, figsize=(18, 16), gridspec_kw={'height_ratios': [2, 1]})  # Adjust size as needed
    # Plot t-SNE on the first subplot
    plt.sca(axs[0])
    # Normalize y-values for color intensity to reflect the magnitude of change
    norm = plt.Normalize(-2.5, 2.5)
    cmap = plt.cm.coolwarm  # Choosing a colormap that spans across negative and positive changes

    lower_thr, upper_thr = -0.5, 0.5

    # Determine the size and alpha dynamically
    sizes = np.where((y > upper_thr) | (y < lower_thr), 50, 12)  # Larger size for rarer values
    alphas = np.where((y > upper_thr) | (y < lower_thr), 1.0, 0.3)  # More opaque for rarer values

    # Ensure sizes and alphas are 1-dimensional arrays
    sizes = sizes.ravel()
    alphas = alphas.ravel()

    # # Scatter plot for all points with varying size and alpha based on change in logIntensity
    # sc = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y, cmap=cmap, norm=norm, s=sizes, alpha=alphas)
    # plt.colorbar(sc, label='Change in logIntensity', extend='both')

    # Sort points by size (or another metric) to ensure larger points are plotted last (on top)
    sort_order = np.argsort(sizes)  # This gives indices that would sort the array

    # Instead of directly indexing with the boolean condition, use it to create a mask and then apply.
    common_points_mask = sizes[sort_order] == 12
    rare_points_mask = sizes[sort_order] == 50

    # Now, apply these masks to the sorted indices to get the correct indices for common and rare points.
    common_points = sort_order[common_points_mask]
    rare_points = sort_order[rare_points_mask]

    # Proceed with your scatter plot as planned
    sc = plt.scatter(
        tsne_result[common_points, 0],
        tsne_result[common_points, 1],
        c=y[common_points],
        cmap=cmap,
        norm=norm,
        s=sizes[common_points],
        alpha=alphas[common_points])

    plt.scatter(
        tsne_result[rare_points, 0],
        tsne_result[rare_points, 1],
        c=y[rare_points],
        cmap=cmap,
        norm=norm,
        s=sizes[rare_points],
        alpha=alphas[rare_points])

    # Add a color bar
    cbar = plt.colorbar(sc, ax=axs[0], label='Change in logIntensity', extend='both')

    # Title and labels
    plt.title(f'{title}\n2D t-SNE Visualization')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')

    # Plot Shepard plot on the second subplot
    plt.sca(axs[1])
    plot_shepard(X, tsne_result)

    # Adjust the subplot layout
    plt.tight_layout()

    # Save the plot
    file_path = f"{prefix}_tsne_plot_{str(save_tag)}.png"
    plt.savefig(file_path)

    if show_plot:
        plt.show()

    plt.close()

    return file_path


def plot_tsne_delta(
        model,
        X: np.ndarray,
        y: np.ndarray,
        title: str,
        prefix: str,
        model_type='features_reg',
        show_plot=False,
        save_tag=None,
        seed=42) -> str:
    """
    Visualizes changes (e.g., in logIntensity) using t-SNE by coloring points based on their values.

    Parameters:
    - model: Trained feature extractor model.
    - X: Input data (NumPy array or compatible).
    - y: Target labels (NumPy array or compatible), representing changes.
    - title: Title for the plot.
    - prefix: Prefix for the file name.
    - model_type: The type of model output to use ('features', 'features_reg', etc.).
    - show_plot: If True, display the plot in addition to saving it.
    - save_tag: Optional tag to append to the file name.
    - seed: Random seed for t-SNE.

    Returns:
    - The file path of the saved t-SNE plot.
    """

    # Extract features based on the model type
    if model_type in ['features_reg_dec', 'features_reg', 'features_dec', 'features_cls']:
        features = model.predict(X)[0]  # Assuming the first output is always features
    elif model_type == 'dict':
        features = model.predict(X)['repr']
    else:
        features = model.predict(X)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=seed)
    tsne_result = tsne.fit_transform(features)

    # Plot setup
    fig, axs = plt.subplots(2, 1, figsize=(18, 16), gridspec_kw={'height_ratios': [2, 1]})  # Adjust size as needed
    # Plot t-SNE on the first subplot
    plt.sca(axs[0])
    # Normalize y-values for color intensity to reflect the magnitude of change
    norm = plt.Normalize(-2.5, 2.5)
    cmap = plt.cm.coolwarm  # Choosing a colormap that spans across negative and positive changes

    lower_thr, upper_thr = -0.5, 0.5

    # Determine the size and alpha dynamically
    sizes = np.where((y > upper_thr) | (y < lower_thr), 50, 12)  # Larger size for rarer values
    alphas = np.where((y > upper_thr) | (y < lower_thr), 1.0, 0.3)  # More opaque for rarer values

    # Ensure sizes and alphas are 1-dimensional arrays
    sizes = sizes.ravel()
    alphas = alphas.ravel()

    # # Scatter plot for all points with varying size and alpha based on change in logIntensity
    # sc = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y, cmap=cmap, norm=norm, s=sizes, alpha=alphas)
    # plt.colorbar(sc, label='Change in logIntensity', extend='both')

    # Sort points by size (or another metric) to ensure larger points are plotted last (on top)
    sort_order = np.argsort(sizes)  # This gives indices that would sort the array

    # Instead of directly indexing with the boolean condition, use it to create a mask and then apply.
    common_points_mask = sizes[sort_order] == 12
    rare_points_mask = sizes[sort_order] == 50

    # Now, apply these masks to the sorted indices to get the correct indices for common and rare points.
    common_points = sort_order[common_points_mask]
    rare_points = sort_order[rare_points_mask]

    # Proceed with your scatter plot as planned
    sc = plt.scatter(
        tsne_result[common_points, 0],
        tsne_result[common_points, 1],
        c=y[common_points],
        cmap=cmap,
        norm=norm,
        s=sizes[common_points],
        alpha=alphas[common_points])

    plt.scatter(
        tsne_result[rare_points, 0],
        tsne_result[rare_points, 1],
        c=y[rare_points],
        cmap=cmap,
        norm=norm,
        s=sizes[rare_points],
        alpha=alphas[rare_points])

    # Add a color bar
    cbar = plt.colorbar(sc, ax=axs[0], label='Change in logIntensity', extend='both')

    # Title and labels
    plt.title(f'{title}\n2D t-SNE Visualization')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')

    # Plot Shepard plot on the second subplot
    plt.sca(axs[1])
    plot_shepard(features, tsne_result)

    # Adjust the subplot layout
    plt.tight_layout()

    # Save the plot
    file_path = f"{prefix}_tsne_plot_{str(save_tag)}.png"
    plt.savefig(file_path)

    if show_plot:
        plt.show()

    plt.close()

    return file_path

def plot_tsne_sarcos(
        model,
        X: np.ndarray,
        y: np.ndarray,
        title: str,
        prefix: str,
        model_type='features_reg',
        lower_threshold=None,
        upper_threshold=None,
        show_plot=False,
        save_tag=None,
        seed=42) -> str:
    """
    Visualizes SARCOS data using t-SNE by coloring and sizing points based on their Torque_1 values.

    Parameters:
    - model: Trained feature extractor model.
    - X: Input data (NumPy array or compatible).
    - y: Target values (Torque_1) (NumPy array or compatible).
    - title: Title for the plot.
    - prefix: Prefix for the file name.
    - model_type: The type of model output to use.
    - lower_threshold: Lower bound threshold for rare values.
    - upper_threshold: Upper bound threshold for rare values.
    - show_plot: If True, display the plot in addition to saving it.
    - save_tag: Optional tag to append to the file name.
    - seed: Random seed for t-SNE.

    Returns:
    - The file path of the saved t-SNE plot.
    """

    # Extract features based on the model type
    if model_type in ['features_reg_dec', 'features_reg', 'features_dec', 'features_cls']:
        features = model.predict(X)[0]  # Assuming the first output is always features
    elif model_type == 'dict':
        features = model.predict(X)['repr']
    else:
        features = model.predict(X)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=seed)
    tsne_result = tsne.fit_transform(features)

    # Plot setup
    fig, axs = plt.subplots(2, 1, figsize=(18, 16), gridspec_kw={'height_ratios': [2, 1]})
    
    # Set up the colormap - using coolwarm as requested
    norm = plt.Normalize(np.min(y), np.max(y))
    cmap = plt.cm.coolwarm  # Using coolwarm colormap as requested

    # Determine the size and alpha dynamically
    is_rare = ((lower_threshold is not None) & (y < lower_threshold)) | \
              ((upper_threshold is not None) & (y > upper_threshold))
    
    sizes = np.where(is_rare, 50, 12)  # Larger size for rare values
    alphas = np.where(is_rare, 1.0, 0.3)  # More opaque for rare values

    # Ensure sizes and alphas are 1-dimensional arrays
    sizes = sizes.ravel()
    alphas = alphas.ravel()

    # Sort points by size to ensure larger points are plotted last (on top)
    sort_order = np.argsort(sizes)
    
    # Separate common and rare points
    common_points_mask = sizes[sort_order] == 12
    rare_points_mask = sizes[sort_order] == 50
    common_points = sort_order[common_points_mask]
    rare_points = sort_order[rare_points_mask]

    # Plot points
    plt.sca(axs[0])
    
    # First plot common points (smaller, more transparent)
    sc = plt.scatter(
        tsne_result[common_points, 0],
        tsne_result[common_points, 1],
        c=y[common_points],
        cmap=cmap,
        norm=norm,
        s=sizes[common_points],
        alpha=alphas[common_points])

    # Then plot rare points (larger, more opaque) to ensure they're on top
    plt.scatter(
        tsne_result[rare_points, 0],
        tsne_result[rare_points, 1],
        c=y[rare_points],
        cmap=cmap,
        norm=norm,
        s=sizes[rare_points],
        alpha=alphas[rare_points])

    # Add a color bar with proper label
    cbar = plt.colorbar(sc, ax=axs[0], label='Torque_1 Value', extend='both')

    # Title
    plt.title(f'{title}\n2D t-SNE Visualization of Torque_1')

    # Plot Shepard plot on the second subplot
    plt.sca(axs[1])
    plot_shepard(features, tsne_result)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    file_path = f"{prefix}_tsne_sarcos_{str(save_tag)}.png"
    plt.savefig(file_path)

    if show_plot:
        plt.show()

    plt.close()

    return file_path


def plot_tsne_onp(
        model,
        X: np.ndarray,
        y: np.ndarray,
        title: str,
        prefix: str,
        model_type='features_reg',
        lower_threshold=None,
        upper_threshold=None,
        show_plot=False,
        save_tag=None,
        seed=42) -> str:
    """
    Visualizes Online News Popularity data using t-SNE by coloring and sizing points based on their popularity values.

    Parameters:
    - model: Trained feature extractor model.
    - X: Input data (NumPy array or compatible).
    - y: Target values (popularity) (NumPy array or compatible).
    - title: Title for the plot.
    - prefix: Prefix for the file name.
    - model_type: The type of model output to use.
    - lower_threshold: Lower bound threshold for rare values.
    - upper_threshold: Upper bound threshold for rare values.
    - show_plot: If True, display the plot in addition to saving it.
    - save_tag: Optional tag to append to the file name.
    - seed: Random seed for t-SNE.

    Returns:
    - The file path of the saved t-SNE plot.
    """

    # Extract features based on the model type
    if model_type in ['features_reg_dec', 'features_reg', 'features_dec', 'features_cls']:
        features = model.predict(X)[0]  # Assuming the first output is always features
    elif model_type == 'dict':
        features = model.predict(X)['repr']
    else:
        features = model.predict(X)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=seed)
    tsne_result = tsne.fit_transform(features)

    # Plot setup
    fig, axs = plt.subplots(2, 1, figsize=(18, 16), gridspec_kw={'height_ratios': [2, 1]})
    
    # Set up the colormap - using coolwarm as requested
    norm = plt.Normalize(np.min(y), np.max(y))
    cmap = plt.cm.coolwarm  # Using coolwarm colormap as requested

    # Determine the size and alpha dynamically
    is_rare = ((lower_threshold is not None) & (y < lower_threshold)) | \
              ((upper_threshold is not None) & (y > upper_threshold))
    
    sizes = np.where(is_rare, 50, 12)  # Larger size for rare values
    alphas = np.where(is_rare, 1.0, 0.3)  # More opaque for rare values

    # Ensure sizes and alphas are 1-dimensional arrays
    sizes = sizes.ravel()
    alphas = alphas.ravel()

    # Sort points by size to ensure larger points are plotted last (on top)
    sort_order = np.argsort(sizes)
    
    # Separate common and rare points
    common_points_mask = sizes[sort_order] == 12
    rare_points_mask = sizes[sort_order] == 50
    common_points = sort_order[common_points_mask]
    rare_points = sort_order[rare_points_mask]

    # Plot points
    plt.sca(axs[0])
    
    # First plot common points (smaller, more transparent)
    sc = plt.scatter(
        tsne_result[common_points, 0],
        tsne_result[common_points, 1],
        c=y[common_points],
        cmap=cmap,
        norm=norm,
        s=sizes[common_points],
        alpha=alphas[common_points])

    # Then plot rare points (larger, more opaque) to ensure they're on top
    plt.scatter(
        tsne_result[rare_points, 0],
        tsne_result[rare_points, 1],
        c=y[rare_points],
        cmap=cmap,
        norm=norm,
        s=sizes[rare_points],
        alpha=alphas[rare_points])

    # Add a color bar with proper label
    cbar = plt.colorbar(sc, ax=axs[0], label='Log Shares', extend='both')

    # Title
    plt.title(f'{title}\n2D t-SNE Visualization of Log Shares')

    # Plot Shepard plot on the second subplot
    plt.sca(axs[1])
    plot_shepard(features, tsne_result)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    file_path = f"{prefix}_tsne_onp_{str(save_tag)}.png"
    plt.savefig(file_path)

    if show_plot:
        plt.show()

    plt.close()

    return file_path

def plot_tsne_asc(
        model,
        X: np.ndarray,
        y: np.ndarray,
        title: str,
        prefix: str,
        model_type='features_reg',
        lower_threshold=None,
        upper_threshold=None,
        show_plot=False,
        save_tag=None,
        seed=42) -> str:
    """
    Visualizes All State Claims data using t-SNE by coloring and sizing points based on their log_cost values.

    Parameters:
    - model: Trained feature extractor model.
    - X: Input data (NumPy array or compatible).
    - y: Target values (log_cost) (NumPy array or compatible).
    - title: Title for the plot.
    - prefix: Prefix for the file name.
    - model_type: The type of model output to use.
    - lower_threshold: Lower bound threshold for rare values.
    - upper_threshold: Upper bound threshold for rare values.
    - show_plot: If True, display the plot in addition to saving it.
    - save_tag: Optional tag to append to the file name.
    - seed: Random seed for t-SNE.

    Returns:
    - The file path of the saved t-SNE plot.
    """

    # Extract features based on the model type
    if model_type in ['features_reg_dec', 'features_reg', 'features_dec', 'features_cls']:
        features = model.predict(X)[0]  # Assuming the first output is always features
    elif model_type == 'dict':
        features = model.predict(X)['repr']
    else:
        features = model.predict(X)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=seed)
    tsne_result = tsne.fit_transform(features)

    # Plot setup
    fig, axs = plt.subplots(2, 1, figsize=(18, 16), gridspec_kw={'height_ratios': [2, 1]})
    
    # Set up the colormap - using coolwarm as requested
    norm = plt.Normalize(np.min(y), np.max(y))
    cmap = plt.cm.coolwarm  # Using coolwarm colormap as requested

    # Determine the size and alpha dynamically
    is_rare = ((lower_threshold is not None) & (y < lower_threshold)) | \
              ((upper_threshold is not None) & (y > upper_threshold))
    
    sizes = np.where(is_rare, 50, 12)  # Larger size for rare values
    alphas = np.where(is_rare, 1.0, 0.3)  # More opaque for rare values

    # Ensure sizes and alphas are 1-dimensional arrays
    sizes = sizes.ravel()
    alphas = alphas.ravel()

    # Sort points by size to ensure larger points are plotted last (on top)
    sort_order = np.argsort(sizes)
    
    # Separate common and rare points
    common_points_mask = sizes[sort_order] == 12
    rare_points_mask = sizes[sort_order] == 50
    common_points = sort_order[common_points_mask]
    rare_points = sort_order[rare_points_mask]

    # Plot points
    plt.sca(axs[0])
    
    # First plot common points (smaller, more transparent)
    sc = plt.scatter(
        tsne_result[common_points, 0],
        tsne_result[common_points, 1],
        c=y[common_points],
        cmap=cmap,
        norm=norm,
        s=sizes[common_points],
        alpha=alphas[common_points])

    # Then plot rare points (larger, more opaque) to ensure they're on top
    plt.scatter(
        tsne_result[rare_points, 0],
        tsne_result[rare_points, 1],
        c=y[rare_points],
        cmap=cmap,
        norm=norm,
        s=sizes[rare_points],
        alpha=alphas[rare_points])

    # Add a color bar with proper label
    cbar = plt.colorbar(sc, ax=axs[0], label='Log Shares', extend='both')

    # Title
    plt.title(f'{title}\n2D t-SNE Visualization of Log Cost')

    # Plot Shepard plot on the second subplot
    plt.sca(axs[1])
    plot_shepard(features, tsne_result)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    file_path = f"{prefix}_tsne_asc_{str(save_tag)}.png"
    plt.savefig(file_path)

    if show_plot:
        plt.show()

    plt.close()

    return file_path

def plot_tsne_blogf(
        model,
        X: np.ndarray,
        y: np.ndarray,
        title: str,
        prefix: str,
        model_type='features_reg',
        freq_threshold=None,
        rare_threshold=None,
        show_plot=False,
        save_tag=None,
        seed=42) -> str:
    """
    Visualizes Blog Feedback data using t-SNE by coloring and sizing points based on their feedback values.
    
    Blog Feedback represents log10 of comments in the next 24 hours with:
    - Frequent values: y < log10(4)
    - Medium values: log10(4) < y < log10(40)
    - Rare values: y > log10(40)

    Parameters:
    - model: Trained feature extractor model.
    - X: Input data (NumPy array or compatible).
    - y: Target values (log10 of comments in next 24h) (NumPy array or compatible).
    - title: Title for the plot.
    - prefix: Prefix for the file name.
    - model_type: The type of model output to use.
    - freq_threshold: Threshold for frequent values (default: log10(4)).
    - rare_threshold: Threshold for rare values (default: log10(40)).
    - show_plot: If True, display the plot in addition to saving it.
    - save_tag: Optional tag to append to the file name.
    - seed: Random seed for t-SNE.

    Returns:
    - The file path of the saved t-SNE plot.
    """

    # Extract features based on the model type
    if model_type in ['features_reg_dec', 'features_reg', 'features_dec', 'features_cls']:
        features = model.predict(X)[0]  # Assuming the first output is always features
    elif model_type == 'dict':
        features = model.predict(X)['repr']
    else:
        features = model.predict(X)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=seed)
    tsne_result = tsne.fit_transform(features)

    # Plot setup
    fig, axs = plt.subplots(2, 1, figsize=(18, 16), gridspec_kw={'height_ratios': [2, 1]})
    
    # Set up the colormap - using coolwarm as requested
    norm = plt.Normalize(np.min(y), np.max(y))
    cmap = plt.cm.coolwarm  # Using coolwarm colormap as requested

    # Determine the size and alpha dynamically based on thresholds
    # Default thresholds if not provided
    if freq_threshold is None:
        freq_threshold = np.log10(4)
    if rare_threshold is None:
        rare_threshold = np.log10(40)
        
    # Define categories: frequent (y < log10(4)), medium (log10(4) < y < log10(40)), rare (y > log10(40))
    is_frequent = y < freq_threshold
    is_rare = y > rare_threshold
    is_medium = ~(is_frequent | is_rare)
    
    # Assign sizes and alphas based on categories
    sizes = np.ones_like(y) * 12  # Default medium size
    sizes[is_frequent] = 30  # Larger for frequent
    sizes[is_rare] = 50  # Largest for rare
    
    alphas = np.ones_like(y) * 0.5  # Default medium alpha
    alphas[is_frequent] = 0.7  # More opaque for frequent
    alphas[is_rare] = 1.0  # Most opaque for rare

    # Ensure sizes and alphas are 1-dimensional arrays
    sizes = sizes.ravel()
    alphas = alphas.ravel()

    # Sort points by size to ensure larger points are plotted last (on top)
    sort_order = np.argsort(sizes)
    
    # Separate points by category for plotting
    frequent_points = np.where(is_frequent)[0]
    medium_points = np.where(is_medium)[0]
    rare_points = np.where(is_rare)[0]

    # Plot points
    plt.sca(axs[0])
    
    # First plot medium points (medium size and transparency)
    sc = plt.scatter(
        tsne_result[medium_points, 0],
        tsne_result[medium_points, 1],
        c=y[medium_points],
        cmap=cmap,
        norm=norm,
        s=sizes[medium_points],
        alpha=alphas[medium_points],
        label=f'Medium ({freq_threshold:.2f} < y < {rare_threshold:.2f})')

    # Then plot frequent points
    plt.scatter(
        tsne_result[frequent_points, 0],
        tsne_result[frequent_points, 1],
        c=y[frequent_points],
        cmap=cmap,
        norm=norm,
        s=sizes[frequent_points],
        alpha=alphas[frequent_points],
        label=f'Frequent (y < {freq_threshold:.2f})')

    # Finally plot rare points (largest, most opaque) to ensure they're on top
    plt.scatter(
        tsne_result[rare_points, 0],
        tsne_result[rare_points, 1],
        c=y[rare_points],
        cmap=cmap,
        norm=norm,
        s=sizes[rare_points],
        alpha=alphas[rare_points],
        label=f'Rare (y > {rare_threshold:.2f})')

    # Add a color bar with proper label
    cbar = plt.colorbar(sc, ax=axs[0], label='Log Comments Next 24h', extend='both')

    # Add legend
    plt.legend()

    # Title
    plt.title(f'{title}\n2D t-SNE Visualization of Blog Feedback (Log Comments Next 24h)')

    # Plot Shepard plot on the second subplot
    plt.sca(axs[1])
    plot_shepard(features, tsne_result)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    file_path = f"{prefix}_tsne_blogf_{str(save_tag)}.png"
    plt.savefig(file_path)

    if show_plot:
        plt.show()

    plt.close()

    return file_path


def plot_tsne_sep(
        model,
        X: np.ndarray,
        y: np.ndarray,
        title: str,
        prefix: str,
        model_type='features_reg',
        show_plot=False,
        save_tag=None,
        seed=42,
        sep_threshold=2.30258509299) -> str:
    """
    Visualizes changes (e.g., in logIntensity) using t-SNE by coloring points based on their values.

    Parameters:
    - model: Trained feature extractor model.
    - X: Input data (NumPy array or compatible).
    - y: Target labels (NumPy array or compatible), representing changes.
    - title: Title for the plot.
    - prefix: Prefix for the file name.
    - model_type: The type of model output to use ('features', 'features_reg', etc.).
    - show_plot: If True, display the plot in addition to saving it.
    - save_tag: Optional tag to append to the file name.
    - seed: Random seed for t-SNE.
    - sep_threshold: Threshold for separating frequent and rare events.

    Returns:
    - The file path of the saved t-SNE plot.
    """

    # Extract features based on the model type
    if model_type in ['features_reg_dec', 'features_reg', 'features_dec', 'features_cls']:
        features = model.predict(X)[0]  # Assuming the first output is always features
    elif model_type == 'dict':
        features = model.predict(X)['repr']
    else:
        features = model.predict(X)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=seed)
    tsne_result = tsne.fit_transform(features)

    # Plot setup
    fig, axs = plt.subplots(2, 1, figsize=(18, 16), gridspec_kw={'height_ratios': [2, 1]})  # Adjust size as needed
    # Plot t-SNE on the first subplot
    plt.sca(axs[0])
    # Normalize y-values for color intensity
    norm = plt.Normalize(min(y), max(y))
    cmap = plt.cm.coolwarm  # Choosing a colormap that spans across negative and positive changes

    # Determine the size and alpha based on sep_threshold
    sizes = np.where(y >= sep_threshold, 50, 12)  # Larger size for rare values (above threshold)
    alphas = np.where(y >= sep_threshold, 1.0, 0.3)  # More opaque for rare values (above threshold)

    # Ensure sizes and alphas are 1-dimensional arrays
    sizes = sizes.ravel()
    alphas = alphas.ravel()

    # Sort points by size to ensure larger points are plotted last (on top)
    sort_order = np.argsort(sizes)  # This gives indices that would sort the array

    # Create masks for frequent and rare points
    frequent_points_mask = sizes[sort_order] == 12
    rare_points_mask = sizes[sort_order] == 50

    # Now, apply these masks to the sorted indices to get the correct indices for frequent and rare points.
    frequent_points = sort_order[frequent_points_mask]
    rare_points = sort_order[rare_points_mask]

    # Proceed with your scatter plot as planned
    sc = plt.scatter(
        tsne_result[frequent_points, 0],
        tsne_result[frequent_points, 1],
        c=y[frequent_points],
        cmap=cmap,
        norm=norm,
        s=sizes[frequent_points],
        alpha=alphas[frequent_points])

    plt.scatter(
        tsne_result[rare_points, 0],
        tsne_result[rare_points, 1],
        c=y[rare_points],
        cmap=cmap,
        norm=norm,
        s=sizes[rare_points],
        alpha=alphas[rare_points])

    # Add a color bar
    cbar = plt.colorbar(sc, ax=axs[0], label='Change in logIntensity', extend='both')

    # Title and labels
    plt.title(f'{title}\n2D t-SNE Visualization')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')

    # Plot Shepard plot on the second subplot
    plt.sca(axs[1])
    plot_shepard(features, tsne_result)

    # Adjust the subplot layout
    plt.tight_layout()

    # Save the plot
    file_path = f"{prefix}_tsne_plot_{str(save_tag)}.png"
    plt.savefig(file_path)

    if show_plot:
        plt.show()

    plt.close()

    return file_path


def investigate_tsne_delta(
        model,
        X: np.ndarray,
        y: np.ndarray,
        title: str,
        prefix: str,
        model_type='features_reg',
        show_plot=False,
        save_tag=None,
        seed=42,
        pred_upper_bound=0.1,
        actual_lower_bound=0.5
) -> str:
    """
    Visualizes changes (e.g., in logIntensity) using t-SNE by coloring points based on their values.

    Parameters:
    - model: Trained feature extractor model.
    - X: Input data (NumPy array or compatible).
    - y: Target labels (NumPy array or compatible), representing changes.
    - title: Title for the plot.
    - prefix: Prefix for the file name.
    - model_type: The type of model output to use ('features', 'features_reg', etc.).
    - show_plot: If True, display the plot in addition to saving it.
    - save_tag: Optional tag to append to the file name.
    - seed: Random seed for t-SNE.
    - pred_upper_bound: Upper threshold for prediction values to highlight.
    - actual_lower_bound: Lower threshold for actual label values to highlight.

    Returns:
    - The file path of the saved t-SNE plot.
    """

    # Ensure model_type is 'features_reg', otherwise, no operation
    if model_type != 'features_reg':
        raise ValueError("Only 'features_reg' model type is supported.")

    # Get features and predictions from the model
    features, predictions = model.predict(X)
    predictions = process_predictions(predictions)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=seed)
    tsne_result = tsne.fit_transform(features)

    # Plot setup
    fig, axs = plt.subplots(2, 1, figsize=(18, 16), gridspec_kw={'height_ratios': [2, 1]})  # Adjust size as needed
    # Plot t-SNE on the first subplot
    plt.sca(axs[0])
    # Normalize y-values for color intensity to reflect the magnitude of change
    norm = plt.Normalize(-2.5, 2.5)
    cmap = plt.cm.coolwarm  # Choosing a colormap that spans across negative and positive changes

    lower_thr, upper_thr = -0.5, 0.5
    # Determine which points to highlight
    highlight_mask = (np.abs(predictions) <= pred_upper_bound) & (np.abs(y) >= actual_lower_bound)

    # Determine the size and alpha dynamically
    sizes = np.where((y > upper_thr) | (y < lower_thr), 50, 12)  # Larger size for rarer values
    alphas = np.where((y > upper_thr) | (y < lower_thr), 1.0, 0.3)  # More opaque for rarer values
    markers = np.where(highlight_mask, 'x', 'o')  # Highlighted points have a different marker

    # Create masks for common and rare points
    common_mask = (y >= lower_thr) & (y <= upper_thr)
    rare_mask = ~common_mask

    # Plotting
    for size, mask, group_name in zip([12, 50], [common_mask, rare_mask], ['common', 'rare']):
        group_mask = mask
        for marker in np.unique(markers[group_mask]):
            specific_mask = group_mask & (markers == marker)
            sc = plt.scatter(
                tsne_result[specific_mask, 0],
                tsne_result[specific_mask, 1],
                c=y[specific_mask],
                cmap=cmap,
                norm=norm,
                s=size,
                alpha=alphas[specific_mask],
                marker=marker,
                label=f'{group_name} {marker}')

    # Add a color bar
    cbar = plt.colorbar(sc, ax=axs[0], label='Change in logIntensity', extend='both')

    # Title and labels
    plt.title(f'{title}\n2D t-SNE Visualization')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    plt.legend()

    # Plot Shepard plot on the second subplot
    plt.sca(axs[1])
    plot_shepard(features, tsne_result)

    # Adjust the subplot layout
    plt.tight_layout()

    # Save the plot
    file_path = f"{prefix}_tsne_plot_{str(save_tag)}.png"
    plt.savefig(file_path)

    if show_plot:
        plt.show()

    plt.close()

    return file_path


def plot_2D_pds(model, X, y, title, prefix, save_tag=None):
    """
    If the feature dimension is 2, this function plots the features in a 2D space. If the feature dimension is not 2,
    it extracts features using the given model and then applies t-SNE for visualization.

    Parameters:
    - model: The trained feature extractor model. If feature_dim is 2, this can be None.
    - X: Input cme_files (NumPy array or compatible)
    - y: Target labels (NumPy array or compatible)
    - title: Title for the plot
    - prefix: Prefix for the file name
    - save_tag: Optional tag to add to the saved filename

    Returns:
    - The path to the saved 2D plot file
    """
    # Define the thresholds
    threshold = np.log(10 / np.exp(2)) + 1e-4
    sep_threshold = np.log(10)

    # Check if feature dimension is 2
    if model is not None:
        features = model.predict(X)
    else:
        features = X

    if features.shape[1] != 2:
        raise ValueError("Feature dimension is not 2, cannot plot directly without t-SNE.")

    # Identify indices based on thresholds
    above_sep_threshold_indices = np.where(y > sep_threshold)[0]
    elevated_event_indices = np.where((y > threshold) & (y <= sep_threshold))[0]
    below_threshold_indices = np.where(y <= threshold)[0]

    plt.figure(figsize=(12, 8))

    # Create scatter plot for below-threshold points (in gray)
    plt.scatter(features[below_threshold_indices, 0], features[below_threshold_indices, 1], marker='o',
                color='gray', alpha=0.6)

    # Normalize y-values for better color mapping
    norm = plt.Normalize(y.min(), y.max())

    # Compute marker sizes based on y-values
    marker_sizes_elevated = 50 * ((y[elevated_event_indices] - y.min()) / (y.max() - y.min())) ** 2 + 10
    marker_sizes_sep = 50 * ((y[above_sep_threshold_indices] - y.min()) / (y.max() - y.min())) ** 2 + 10

    # Create scatter plot for elevated events (circle marker)
    plt.scatter(features[elevated_event_indices, 0], features[elevated_event_indices, 1],
                c=y[elevated_event_indices], cmap='plasma', norm=norm, alpha=0.6, marker='o', s=marker_sizes_elevated)

    # Create scatter plot for SEPs (diamond marker)
    plt.scatter(features[above_sep_threshold_indices, 0], features[above_sep_threshold_indices, 1],
                c=y[above_sep_threshold_indices], cmap='plasma', norm=norm, alpha=0.6, marker='d', s=marker_sizes_sep)

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Label Value')

    # Add a legend
    legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10),
                     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
                     plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='red', markersize=10)]

    plt.legend(legend_labels, ['Background', 'Elevated Events', 'SEPs'],
               loc='upper left')

    plt.title(f'{title}\n2D Feature Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Save the plot
    file_path = f"{prefix}_2d_features_plot_{save_tag}.png"
    plt.savefig(file_path)
    plt.close()

    return file_path


def count_above_threshold(y_values: List[float], threshold: float = 0.3027, sep_threshold: float = 2.3026) -> tuple[
    bool_, bool_]:
    """
    Count the number of y values that are above a given threshold for elevated events and above a sep_threshold for sep events.

    Parameters:
    - y_values (List[float]): The array of y-values to check.
    - threshold (float, optional): The threshold value to use for counting elevated events. Default is log(10 / e^2).
    - sep_threshold (float, optional): The threshold value to use for counting sep events. Default is log(10).

    Returns:
    - Tuple[int, int]: The count of y-values that are above the thresholds for elevated and sep events, respectively.
    """

    y_values_np = np.array(y_values)  # Convert list to NumPy array
    elevated_count = np.sum((y_values_np > threshold) & (y_values_np <= sep_threshold))
    sep_count = np.sum(y_values_np > sep_threshold)

    return elevated_count, sep_count
