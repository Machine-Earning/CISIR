from datetime import datetime
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
# from sklearn.manifold import TSNE
from tsnecuda import TSNE
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde
from modules.evaluate import evaluation as eval
from modules.training import seploader as sepl


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
    scatter = plt.scatter(distances_target_norm, distances_repr_norm, c=z, s=15, edgecolor='', cmap='viridis', alpha=0.5)
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
    if model_type in ['features_reg_dec', 'features_reg', 'features_dec']:
        features = model.predict(X)[0]  # Assuming the first output is always features
    else:  # model_type == 'features'
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


# def plot_tsne_extended(
#         model, X, y,
#         title, prefix,
#         model_type='features_reg',
#         threshold: float = None, sep_threshold: float = None,
#         show_plot=False,
#         save_tag=None, seed=42):
#     """
#     Applies t-SNE to the features extracted by the given model and saves the plot in 2D with a timestamp.
#     The color of the points is determined by their label values.
#
#     Parameters:
#     - model: Trained feature extractor model
#     - X: Input cme_files (NumPy array or compatible)
#     - y: Target labels (NumPy array or compatible)
#     - title: Title for the plot
#     - prefix: Prefix for the file name
#     - sep_shape: The shape of the marker to use for SEP events (above threshold).
#     - model_type: The type of model to use (feature, feature_reg, features_reg_dec)
#     - threshold: Threshold for elevated events
#     - sep_threshold: Threshold for SEP events
#     - save_tag: Optional tag to add to the saved file name
#     - seed: Random seed for t-SNE
#
#
#     Returns:
#     - Saves a 2D t-SNE plot to a file with a timestamp
#     """
#     # T = 1.57381089527  # 0.4535
#     # # Define the thresholds
#     # if threshold is None:
#     #     threshold = np.log(T / np.exp(2)) + 1e-4
#     # if sep_threshold is None:
#     #     sep_threshold = np.log(T)
#
#     # threshold = np.log(10 / np.exp(2)) + 1e-4
#     # sep_threshold = np.log(10)
#
#     # Extract features using the trained extended model
#     if model_type == 'features_reg_dec':
#         features, _, _ = model.predict(X)
#     elif model_type == 'features_reg' or model_type == 'features_dec':
#         features, _ = model.predict(X)
#     else:  # model_type == 'features'
#         features = model.predict(X)
#
#     # Apply t-SNE
#     tsne = TSNE(n_components=2, random_state=seed)
#     tsne_result = tsne.fit_transform(features)
#
#     # # Identify indices based on thresholds
#     # above_sep_threshold_indices = np.where(y > sep_threshold)[0]
#     # elevated_event_indices = np.where((y > threshold) & (y <= sep_threshold))[0]
#     # below_threshold_indices = np.where(y <= threshold)[0]
#
#     # plt.figure(figsize=(12, 8))
#     # Adjusted subplot layout
#     fig, axs = plt.subplots(2, 1, figsize=(18, 16), gridspec_kw={'height_ratios': [2, 1]})  # Adjust size as needed
#
#     # Plot t-SNE on the first subplot
#     plt.sca(axs[0])
#     # Create scatter plot for below-threshold points (in gray)
#     plt.scatter(tsne_result[below_threshold_indices, 0], tsne_result[below_threshold_indices, 1], marker='o',
#                 color='gray', alpha=0.6)
#
#     # Normalize y-values for better color mapping
#     norm = plt.Normalize(y.min(), y.max())
#
#     # Compute marker sizes based on y-values. Squaring to amplify differences.
#     marker_sizes_elevated = 50 * ((y[elevated_event_indices] - y.min()) / (y.max() - y.min())) ** 2 + 10
#     marker_sizes_sep = 50 * ((y[above_sep_threshold_indices] - y.min()) / (y.max() - y.min())) ** 2 + 10
#
#     # Create scatter plot for elevated events (square marker)
#     plt.scatter(tsne_result[elevated_event_indices, 0], tsne_result[elevated_event_indices, 1],
#                 c=y[elevated_event_indices], cmap='plasma', norm=norm, alpha=0.6, marker='o', s=marker_sizes_elevated)
#
#     # Create scatter plot for SEPs (diamond marker)
#     plt.scatter(tsne_result[above_sep_threshold_indices, 0], tsne_result[above_sep_threshold_indices, 1],
#                 c=y[above_sep_threshold_indices], cmap='plasma', norm=norm, alpha=0.6, marker='d', s=marker_sizes_sep)
#
#     # Add a color bar
#     sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm)
#     cbar.set_label('ln Intensity')
#
#     # Add a legend
#     legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10),
#                      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
#                      plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='red', markersize=10)]
#
#     plt.legend(legend_labels, ['Background', 'Elevated Events (darker colors)', 'SEPs (lighter colors)'],
#                loc='upper left')
#
#     plt.title(f'{title}\n2D t-SNE Visualization')
#     # plt.xlabel('Dimension 1')
#     # plt.ylabel('Dimension 2')
#
#     # Plot Shepard plot on the second subplot
#     plt.sca(axs[1])
#     plot_shepard(features, tsne_result)
#
#     # Adjust the subplot layout
#     plt.tight_layout()
#
#     # Save the plot
#     file_path = f"{prefix}_tsne_plot_{str(save_tag)}.png"
#     plt.savefig(file_path)
#
#     if show_plot:
#         plt.show()
#
#     plt.close()
#
#     return file_path
#
# def plot_tsne_pds(model, X, y, title, prefix, save_tag=None, seed=42):
#     """
#     Applies t-SNE to the features extracted by the given model and saves the plot in 2D with a timestamp.
#     The color of the points is determined by their label values.
#
#     Parameters:
#     - model: Trained feature extractor model
#     - X: Input cme_files (NumPy array or compatible)
#     - y: Target labels (NumPy array or compatible)
#     - prefix: Prefix for the file name
#
#     Returns:
#     - Saves a 2D t-SNE plot to a file with a timestamp
#     """
#     T = 1.57381089527  # 0.4535
#     # Define the thresholds
#     threshold = np.log(T / np.exp(2)) + 1e-4
#     sep_threshold = np.log(T)
#
#     # Extract features using the trained model
#     features = model.predict(X)
#
#     # Apply t-SNE
#     tsne = TSNE(n_components=2, random_state=seed)
#     tsne_result = tsne.fit_transform(features)
#
#     # Identify indices based on thresholds
#     above_sep_threshold_indices = np.where(y > sep_threshold)[0]
#     elevated_event_indices = np.where((y > threshold) & (y <= sep_threshold))[0]
#     below_threshold_indices = np.where(y <= threshold)[0]
#
#     # plt.figure(figsize=(12, 8))
#     # Adjusted subplot layout
#     fig, axs = plt.subplots(2, 1, figsize=(18, 16), gridspec_kw={'height_ratios': [2, 1]})  # Adjust size as needed
#
#     # Plot t-SNE on the first subplot
#     plt.sca(axs[0])
#     # Create scatter plot for below-threshold points (in gray)
#     plt.scatter(tsne_result[below_threshold_indices, 0],
#                 tsne_result[below_threshold_indices, 1],
#                 marker='o',
#                 color='gray', alpha=0.6)
#
#     # Normalize y-values for better color mapping
#     norm = plt.Normalize(y.min(), y.max())
#
#     # Compute marker sizes based on y-values
#     marker_sizes_elevated = 50 * ((y[elevated_event_indices] - y.min()) / (y.max() - y.min())) ** 2 + 10
#     marker_sizes_sep = 50 * ((y[above_sep_threshold_indices] - y.min()) / (y.max() - y.min())) ** 2 + 10
#
#     # Create scatter plot for elevated events (square marker)
#     plt.scatter(tsne_result[elevated_event_indices, 0],
#                 tsne_result[elevated_event_indices, 1],
#                 c=y[elevated_event_indices],
#                 cmap='plasma',
#                 norm=norm,
#                 alpha=0.6,
#                 marker='o',
#                 s=marker_sizes_elevated)
#
#     # Create scatter plot for SEPs (diamond marker)
#     plt.scatter(tsne_result[above_sep_threshold_indices, 0],
#                 tsne_result[above_sep_threshold_indices, 1],
#                 c=y[above_sep_threshold_indices],
#                 cmap='plasma',
#                 norm=norm,
#                 alpha=0.6,
#                 marker='d',
#                 s=marker_sizes_sep)
#
#     # Add a color bar
#     sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm)
#     cbar.set_label('Label Value')
#
#     # Add legend
#     # Add a legend
#     legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10),
#                      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
#                      plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='red', markersize=10)]
#
#     plt.legend(legend_labels,
#                ['Background', 'Elevated Events (darker colors)', 'SEPs (lighter colors)'],
#                loc='upper left')
#
#     plt.title(f'{title}\nT-SNE Visualization')
#     # plt.xlabel('Dimension 1')
#     # plt.ylabel('Dimension 2')
#
#     print("Done with t-sne, starting with shepard...")
#
#     # Plot Shepard plot on the second subplot
#     plt.sca(axs[1])
#     plot_shepard(features, tsne_result)
#
#     # Adjust the subplot layout
#     plt.tight_layout()
#
#     # Save the plot
#     file_path = f"{prefix}_tsne_plot_{str(save_tag)}.png"
#     plt.savefig(file_path)
#     plt.close()
#
#     return file_path


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


def count_above_threshold(y_values: List[float], threshold: float = 0.3027, sep_threshold: float = 2.3026) -> Tuple[
    int, int]:
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


def load_model_with_weights(model_type: str,
                            weight_path: str,
                            input_dim: int = 19,
                            feat_dim: int = 9,
                            hiddens: Optional[list] = None,
                            output_dim: int = 1) -> tf.keras.Model:
    """
    Load a model of a given type with pre-trained weights.

    :param output_dim:
    :param model_type: The type of the model to load ('features', 'reg', 'dec', 'features_reg_dec', etc.).
    :param weight_path: The path to the saved weights.
    :param input_dim: The input dimension for the model. Default is 19.
    :param feat_dim: The feature dimension for the model. Default is 9.
    :param hiddens: A list of integers specifying the number of hidden units for each hidden layer.
    :return: A loaded model with pre-trained weights.
    """
    if hiddens is None:
        hiddens = [18]
    mb = modeling.ModelBuilder()

    model = None  # will be determined by model type
    if model_type == 'features_reg_dec':
        # Add logic to create this type of model
        model = mb.create_model_pds(
            input_dim=input_dim,
            feat_dim=feat_dim,
            hiddens=hiddens,
            output_dim=output_dim,
            with_ae=True, with_reg=True)
    elif model_type == 'features_reg':
        model = mb.create_model_pds(
            input_dim=input_dim,
            feat_dim=feat_dim,
            hiddens=hiddens,
            output_dim=output_dim,
            with_ae=False, with_reg=True)
    elif model_type == 'features_dec':
        model = mb.create_model_pds(
            input_dim=input_dim,
            feat_dim=feat_dim,
            hiddens=hiddens,
            output_dim=None,
            with_ae=True, with_reg=False)
    else:  # features
        model = mb.create_model_pds(
            input_dim=input_dim,
            feat_dim=feat_dim,
            hiddens=hiddens,
            output_dim=None,
            with_ae=False, with_reg=False)
    # Load weights into the model
    model.load_weights(weight_path)
    print(f"Weights {weight_path} loaded successfully!")

    return model


def load_and_plot_tsne(
        model_path,
        model_type,
        title,
        data_dir='/home1/jmoukpe2016/keras-functional-api/cme_files/fold/fold_1',
        with_head=False):
    """
    Load a trained model and plot its t-SNE visualization.

    :param with_head:
    :param model_path: Path to the saved model.
    :param model_type: Type of model to load ('features', 'features_reg', 'features_reg_dec').
    :param title: Title for the t-SNE plot.
    :param sep_marker: The shape of the marker to use for SEP events (above threshold).
    :param data_dir: Directory where the cme_files files are stored.
    """

    # print the parameters
    print('Model type:', model_type)
    print('Model path:', model_path)
    print('Title:', title)
    print('Data directory:', data_dir)

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # check for gpus
    print(tf.config.list_physical_devices('GPU'))
    # Load the appropriate model
    mb = modeling.ModelBuilder()

    if with_head:
        if model_type == 'features':
            features_model = mb.create_model_pds(input_dim=19, feat_dim=9, hiddens=[18])
            loaded_model = mb.add_reg_proj_head(features_model, freeze_features=False, pds=True)
        elif model_type == 'features_reg':
            features_model = mb.create_model(input_dim=19, feat_dim=9, output_dim=1, hiddens=[18])
            loaded_model = mb.add_reg_proj_head(features_model, freeze_features=False)
        elif model_type == 'features_reg_dec':
            features_model = mb.create_model(input_dim=19, feat_dim=9, output_dim=1, hiddens=[18], with_ae=True)
            loaded_model = mb.add_reg_proj_head(features_model, freeze_features=False)
        else:  # regular reg
            loaded_model = mb.create_model(input_dim=19, feat_dim=9, output_dim=1, hiddens=[18])

        loaded_model.load_weights(model_path)
        print(f'Model loaded from {model_path}')
    else:
        # load weights to continue training
        loaded_model = load_model_with_weights(model_type, model_path)

    # Load cme_files
    loader = sepl.SEPLoader()
    train_x, train_y, val_x, val_y, test_x, test_y = loader.load_from_dir(data_dir)
    # Extract counts of events
    elevateds, seps = count_above_threshold(train_y)
    print(f'Sub-Training set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(val_y)
    print(f'Validation set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(test_y)
    print(f'Test set: elevated events: {elevateds}  and sep events: {seps}')

    # Combine training and validation sets
    combined_train_x, combined_train_y = loader.combine(train_x, train_y, val_x, val_y)

    # Plot and save t-SNE
    test_plot_path = plot_tsne_extended(loaded_model,
                                        test_x, test_y,
                                        title,
                                        model_type + '_testing_',
                                        model_type=model_type,
                                        save_tag=timestamp)

    training_plot_path = plot_tsne_extended(loaded_model,
                                            combined_train_x,
                                            combined_train_y,
                                            title,
                                            model_type + '_training_',
                                            model_type=model_type,
                                            save_tag=timestamp)

    return test_plot_path, training_plot_path


def load_and_test(model_path, model_type, title, threshold=10,
                  data_dir='/home1/jmoukpe2016/keras-functional-api/cme_files/fold/fold_1'):
    """
    Load a trained model and evaluate its performance.

    :param model_path: Path to the saved model.
    :param model_type: Type of model to load ('features', 'features_reg', 'features_reg_dec').
    :param title: Title for the evaluation results.
    :param threshold: Threshold for evaluation.
    :param data_dir: Directory where the cme_files files are stored.
    """

    # print the parameters
    print('Model type:', model_type)
    print('Model path:', model_path)
    print('Title:', title)
    print('Threshold:', threshold)
    print('Data directory:', data_dir)

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # check for gpus
    print(tf.config.list_physical_devices('GPU'))
    # Load the appropriate model
    mb = modeling.ModelBuilder()

    if model_type == 'features':
        features_model = mb.create_model_pds(input_dim=19, feat_dim=9, hiddens=[18])
        loaded_model = mb.add_reg_proj_head(features_model, freeze_features=False)
    elif model_type == 'features_reg':
        features_model = mb.create_model(input_dim=19, feat_dim=9, output_dim=1, hiddens=[18])
        loaded_model = mb.add_reg_proj_head(features_model, freeze_features=False)
    else:  # features_reg_dec
        features_model = mb.create_model(input_dim=19, feat_dim=9, output_dim=1, hiddens=[18], with_ae=True)
        loaded_model = mb.add_reg_proj_head(features_model, freeze_features=False)

    loaded_model.load_weights(model_path)
    print(f'Model loaded from {model_path}')

    # Load cme_files
    loader = sepl.SEPLoader()
    train_x, train_y, val_x, val_y, test_x, test_y = loader.load_from_dir(data_dir)
    # Extract counts of events
    elevateds, seps = count_above_threshold(train_y)
    print(f'Sub-Training set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(val_y)
    print(f'Validation set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(test_y)
    print(f'Test set: elevated events: {elevateds}  and sep events: {seps}')

    # Combine training and validation sets
    combined_train_x, combined_train_y = loader.combine(train_x, train_y, val_x, val_y)

    # Evaluate and save results
    ev = eval.Evaluator()
    ev.evaluate(loaded_model, test_x, test_y, title, threshold=threshold, save_tag=model_type + '_test_' + timestamp)
    ev.evaluate(loaded_model, combined_train_x, combined_train_y, title, threshold=threshold,
                save_tag=model_type + '_training_' + timestamp)
