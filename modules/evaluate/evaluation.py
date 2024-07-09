##############################################################################################################
# Description: this file will be used for evaluation (metrics, plots, experiments,  ...)
# generally each configuration is run 5 times to smooth out different results from random initialization.
##############################################################################################################


import logging
import math
# types for type hinting
from typing import Tuple, List, Optional, Any, Dict, Union

import matplotlib.pyplot as plt
# imports
import numpy as np
import tensorflow as tf
from numpy import ndarray
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras import Model

from modules.training.cme_modeling import error


def find_k_nearest_neighbors(
        X_test: np.ndarray,
        y_test: np.ndarray,
        predictions: np.ndarray,
        k_neighbors: int = 3,
        threshold: float = 0.5,
        max_samples: Optional[int] = None,
        log_results: bool = True,
        logger: Optional[logging.Logger] = None
) -> List[Tuple[int, float, float, List[Tuple[float, int, float, float]]]]:
    """
    Find the k nearest neighbors for each point in the test set with target labels greater than the threshold.
    Results are sorted in ascending order of the test points' actual target labels before logging.

    Args:
        X_test (np.ndarray): Test data features.
        y_test (np.ndarray): Test data target labels (1D array).
        predictions (np.ndarray): Model predictions for the test data (1D array).
        k_neighbors (int): Number of nearest neighbors to find.
        threshold (float): Threshold for selecting positive samples.
        max_samples (Optional[int]): Maximum number of samples to process.
        log_results (bool): Whether to log the results.
        logger (Optional[logging.Logger]): Logger object to use. If None, a new logger will be created.

    Returns:
        List[Tuple[int, float, float, List[Tuple[float, int, float, float]]]]: Sorted list of tuples containing:
            - Test point index
            - Actual target label
            - Predicted label
            - List of tuples for each neighbor:
                - Distance
                - Neighbor index
                - Neighbor's target label
                - Neighbor's predicted label
    """
    if k_neighbors >= len(X_test):
        raise ValueError(f"k ({k_neighbors}) must be less than the number of test samples ({len(X_test)})")

    if len(X_test) != len(y_test) or len(X_test) != len(predictions):
        raise ValueError("Inconsistent lengths of input arrays")

    if log_results and logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    # k+1 because the point itself is included
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto').fit(X_test)

    y_test = y_test.flatten()
    predictions = predictions.flatten()

    large_positives_indices = np.where(y_test > threshold)[0]
    if max_samples:
        large_positives_indices = large_positives_indices[:max_samples]

    if log_results:
        logger.info(f"Processing {len(large_positives_indices)} samples with target labels > {threshold}")

    results = []
    for idx in large_positives_indices:
        distances, indices = nbrs.kneighbors([X_test[idx]])
        # Remove the first neighbor (which is the point itself)
        neighbors = [
            (float(dist), int(neighbor_idx), float(y_test[neighbor_idx]), float(predictions[neighbor_idx]))
            for dist, neighbor_idx in zip(distances[0][1:], indices[0][1:])
        ]
        result = (int(idx), float(y_test[idx]), float(predictions[idx]), neighbors)
        results.append(result)

    # Sort results based on the actual target labels of test points
    sorted_results = sorted(results, key=lambda x: x[1])

    if log_results:
        logger.info(f"Processed and sorted {len(sorted_results)} samples")
        logger.info("Results sorted in ascending order of test points' actual target labels")
        for idx, true_label, pred_label, neighbors in sorted_results:
            logger.info(f"Test point {idx}: true={true_label:.2f}, pred={pred_label:.2f}")
            for i, (dist, neighbor_idx, neighbor_true, neighbor_pred) in enumerate(neighbors):
                logger.info(
                    f"  Neighbor {i + 1}: idx={neighbor_idx}, dist={dist:.4f}, true={neighbor_true:.2f}, pred={neighbor_pred:.2f}")

    return sorted_results


def pds_loss_eval(y_true, z_pred, reduction='none'):
    """
    Computes the loss for a batch of predicted features and their labels.

    :param y_true: A batch of true label values, shape of [batch_size, 1].
    :param z_pred: A batch of predicted Z values, shape of [batch_size, 2].
    :param reduction: The type of reduction to apply to the loss ('sum', 'none', or 'mean').
    :return: The average error for all unique combinations of the samples in the batch.
    """
    int_batch_size = len(z_pred)
    total_error = 0.0

    # print("received batch size in custom eval:", int_batch_size)

    # Loop through all unique pairs of samples in the batch
    for i in range(int_batch_size):
        for j in range(i + 1, int_batch_size):
            z1, z2 = z_pred[i], z_pred[j]
            label1, label2 = y_true[i], y_true[j]
            # Update pair counts (implement this function as needed)
            # update_pair_counts(label1, label2)
            err = error(z1, z2, label1, label2)  # Make sure 'error' function uses NumPy or standard Python
            total_error += err

    if reduction == 'sum':
        return total_error  # total loss
    elif reduction == 'none' or reduction == 'mean':
        denom = int_batch_size * (int_batch_size - 1) / 2 + 1e-9
        return total_error / denom  # average loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}.")


def pds_loss_eval_pairs(y_true, z_pred, reduction='none'):
    """
    Computes the loss for a batch of predicted features and their labels.
    Returns a dictionary of average losses for each pair type and overall.

    :param y_true: A batch of true label values, shape of [batch_size, 1].
    :param z_pred: A batch of predicted Z values, shape of [batch_size, 2].
    :param reduction: The type of reduction to apply to the loss ('sum', 'none', or 'mean').
    :return: A dictionary containing the average errors for all pair types and overall.
    """
    int_batch_size = len(z_pred)
    total_error = 0.0
    pair_errors = {
        'sep_sep': 0.0,
        'sep_elevated': 0.0,
        'sep_background': 0.0,
        'elevated_elevated': 0.0,
        'elevated_background': 0.0,
        'background_background': 0.0
    }
    pair_counts = {key: 0 for key in pair_errors.keys()}

    # print("Received batch size in custom eval:", int_batch_size)

    # Loop through all unique pairs of samples in the batch
    for i in range(int_batch_size):
        for j in range(i + 1, int_batch_size):
            z1, z2 = z_pred[i], z_pred[j]
            label1, label2 = y_true[i], y_true[j]

            # Determine the pair type
            pair_type = determine_pair_type(label1, label2)  # Implement this function
            err = error(z1, z2, label1, label2)  # Make sure 'error' function uses NumPy or standard Python
            pair_errors[pair_type] += err
            pair_counts[pair_type] += 1
            total_error += err

    # Apply reduction
    if reduction == 'sum':
        avg_pair_errors = {key: error_sum for key, error_sum in pair_errors.items()}
        avg_pair_errors['overall'] = total_error
    elif reduction == 'none' or reduction == 'mean':
        avg_pair_errors = {key: pair_errors[key] / pair_counts[key] if pair_counts[key] > 0 else 0 for key in
                           pair_errors}
        denom = int_batch_size * (int_batch_size - 1) / 2 + 1e-9
        avg_pair_errors['overall'] = total_error / denom
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}.")

    return avg_pair_errors


def determine_pair_type(label1, label2, sep_threshold=None, elevated_threshold=None):
    """
    Determines the pair type based on the labels.

    :param label1: The label of the first sample.
    :param label2: The label of the second sample.
    :param sep_threshold: The threshold to classify SEP samples.
    :param elevated_threshold: The threshold to classify elevated samples.
    :return: A string representing the pair type.
    """

    if sep_threshold is None:
        sep_threshold = np.log(10)

    if elevated_threshold is None:
        elevated_threshold = np.log(10.0 / np.exp(2))

    if label1 > sep_threshold and label2 > sep_threshold:
        return 'sep_sep'
    elif (label1 > sep_threshold and label2 > elevated_threshold) or (
            label2 > sep_threshold and label1 > elevated_threshold):
        return 'sep_elevated'
    elif (label1 > sep_threshold and label2 <= elevated_threshold) or (
            label2 > sep_threshold and label1 <= elevated_threshold):
        return 'sep_background'
    elif label1 > elevated_threshold and label2 > elevated_threshold:
        return 'elevated_elevated'
    elif (label1 > elevated_threshold >= label2) or (
            label2 > elevated_threshold >= label1):
        return 'elevated_background'
    else:
        return 'background_background'


def evaluate(model, X, y, batch_size=-1, pairs=False):
    """
    Custom evaluate function to compute loss over the dataset.

    :param model: The trained model.
    :param X: Input features.
    :param y: True labels.
    :param batch_size: Size of the batch, use the whole dataset if batch_size <= 0.
    :param pairs: If True, uses pds_loss_eval_pairs to evaluate loss on pairs.
    :return: Calculated loss over the dataset or a dictionary of losses for each pair type.
    """
    if batch_size <= 0:
        z_pred = model.predict(X)
        return pds_loss_eval_pairs(y, z_pred) if pairs else pds_loss_eval(y, z_pred)

    total_loss = 0
    pair_losses = {key: 0.0 for key in
                   ['sep_sep', 'sep_elevated', 'sep_background', 'elevated_elevated', 'elevated_background',
                    'background_background']}
    pair_counts = {key: 0 for key in pair_losses}
    total_batches = 0

    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        z_pred = model.predict(X_batch)

        if pairs:
            batch_pair_losses = pds_loss_eval_pairs(y_batch, z_pred)
            for key in batch_pair_losses:
                pair_losses[key] += batch_pair_losses[key]
                pair_counts[key] += 1  # Count each batch for each pair type
        else:
            total_loss += pds_loss_eval(y_batch, z_pred)

        total_batches += 1

    if pairs:
        # Compute average losses for each pair type
        avg_pair_losses = {key: pair_losses[key] / pair_counts[key] if pair_counts[key] > 0 else 0 for key in
                           pair_losses}
        return avg_pair_losses

    return total_loss / total_batches if total_batches > 0 else 0


# def evaluate(model, X, y, batch_size=-1):
#     """
#     Custom evaluate function to compute loss over the dataset.
#
#     :param model: The trained model.
#     :param X: Input features.
#     :param y: True labels.
#     :param batch_size: Size of the batch, use the whole dataset if batch_size <= 0.
#     :return: Calculated loss over the dataset.
#     """
#     total_loss = 0
#     total_batches = 0
#
#     # print batch size received
#     print(f'batch size received: {batch_size}')
#
#     if batch_size <= 0:
#         # Use the whole dataset
#         z_pred = model.predict(X)
#         total_loss = pds_loss_eval(y, z_pred, reduction='none')
#         total_batches = 1
#     else:
#         # Process in batches
#         for i in range(0, len(X), batch_size):
#             X_batch = X[i:i + batch_size]
#             y_batch = y[i:i + batch_size]
#             z_pred = model.predict(X_batch)  # model prediction
#             batch_loss = pds_loss_eval(y_batch, z_pred, reduction='none')
#             total_loss += batch_loss
#             total_batches += 1
#
#     average_loss = total_loss / total_batches
#     return average_loss


# Helper function to map 2D indices to 1D indices (assuming it's defined elsewhere in your code)
# def map_to_1D_idx(i, j, n):
#     return n * i + j

class Evaluator:
    """
    Class for evaluating the performance of a model.
    """

    # class variables
    debug = False
    min_bin_y = None
    max_bin_y = None
    bin_size = None
    bins = None
    num_bins = None

    def __init__(self):
        pass

    def evaluate(self, model: Model, X_test: np.ndarray, y_test: np.ndarray, title, res: float = 0.5,
                 threshold: float = 10, save_tag=None) -> Dict[str, Union[float, Any]]:
        """
        Evaluate the performance of the model on test cme_files using TensorFlow's MSE and plot error per bin.

        :param save_tag:
        :param title:
        :param threshold:
        :param model: Model to test
        :param X_test: Test features as a NumPy array.
        :param y_test: Test labels for the regression output as a NumPy array.
        :param res: The resolution of the bins for plotting error per bin.
        :return: Performance as a percentage based on MSE. Lower is better.
        """

        threshold_val = threshold
        threshold = np.log(threshold_val)

        # Predict the y-values using the model
        y_pred = model.predict(X_test)

        # Assuming y_pred may have multiple outputs and you're interested in the regression head
        if isinstance(y_pred, list) and len(y_pred) > 1:
            if self.debug:
                print(f"y_pred: {y_pred}")
            y_pred = y_pred[1]

        # Flatten arrays for easier calculations
        y_pred = y_pred.flatten()
        y_test = y_test.flatten()

        # Calculate the Mean Squared Error using TensorFlow
        mae = tf.keras.losses.MeanAbsoluteError()(y_test, y_pred).numpy()

        # Calculate the Root Mean Squared Error
        # rmse = np.sqrt(mse)

        # Print the MSE and RMSE
        print(f"{save_tag} Mean Absolute Error: {mae}")
        # print(f"Root Mean Squared Error: {rmse}")

        # # Generate bins for plotting
        self.get_bins(list(y_test), res)
        #
        # # Plot error per bin
        # self.plot_error_per_bin(y_test, y_pred, save_tag=save_tag)

        # Define lower threshold
        lower_threshold = np.log(threshold_val / np.exp(2)) + 1e-4  # + 1e-9 to avoid backgrounds being considered

        # Identify different types of events
        SEP_indices = np.where(y_test > threshold)[0]
        Elevated_indices = np.where((y_test <= threshold) & (y_test > lower_threshold))[0]
        Background_indices = np.where(y_test <= lower_threshold)[0]

        y_test_SEP = y_test[SEP_indices]
        y_pred_SEP = y_pred[SEP_indices]

        # Calculate the MAE for only SEP events
        mae_SEP = tf.keras.losses.MeanAbsoluteError()(y_test_SEP, y_pred_SEP).numpy()
        print(f"{save_tag} Mean Absolute Error for SEP events: {mae_SEP}")

        # Classify events based on the given threshold
        y_true = (y_test > threshold).astype(int)
        y_pred_class = (y_pred > threshold).astype(int)

        # Calculate the confusion matrix
        TN, FP, FN, TP = confusion_matrix(y_true, y_pred_class).ravel()

        # Calculate F1 score
        f1 = f1_score(y_true, y_pred_class)

        # Calculate True Skill Statistic (TSS)
        TSS = TP / (TP + FN) - FP / (FP + TN)

        # Calculate Heidke Skill Score (HSS)
        HSS = 2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))

        print(f"True Positives: {TP}, False Positives: {FP}, True Negatives: {TN}, False Negatives: {FN}")
        print(f"F1 Score: {f1}")
        print(f"True Skill Statistic (TSS): {TSS}")
        print(f"Heidke Skill Score (HSS): {HSS}")
        print('-----------------------------------------------------')

        # Plot actual vs predicted
        plt.figure(figsize=(10, 10))
        plt.scatter(y_test[Background_indices], y_pred[Background_indices], c='b', alpha=0.5, label='Background')
        plt.scatter(y_test[Elevated_indices], y_pred[Elevated_indices], c='g', alpha=0.5, label='Elevated')
        plt.scatter(y_test[SEP_indices], y_pred[SEP_indices], c='r', alpha=0.5, label='SEP')
        plt.axhline(threshold, color='gray', linestyle='--', linewidth=0.8)
        plt.axvline(threshold, color='gray', linestyle='--', linewidth=0.8)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'gray', linestyle='--', linewidth=0.8)
        plt.xlabel('Actual Ln Peak Intensity')
        plt.ylabel('Predicted Ln Peak Intensity')
        plt.title(title)
        plt.legend()
        # Save the plot
        file_path = f"test_{threshold_val}_matrix_plot_{str(save_tag)}.png"
        plt.savefig(file_path)
        plt.close()

        # Calculate individual errors
        individual_errors = np.abs(y_test - y_pred)

        # Find indices of the largest 5 errors
        largest_errors_idx = np.argsort(individual_errors)[-5:]

        # Print the bins and errors of these samples
        for idx in largest_errors_idx:
            error = individual_errors[idx]
            y_value = y_test[idx]
            bin_number = np.digitize(y_value, self.bins)
            print(f"Sample with y = {y_value} belongs to bin {bin_number} and has an error of {error}")

        metrics = {'MAE': mae, 'MAE_SEP': mae_SEP, 'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN, 'F1_Score': f1, 'TSS': TSS,
                   'HSS': HSS, 'plot': file_path}
        # Store the metrics in the dictionary

        return metrics

    def plot_error_per_bin(self, y_true: np.ndarray, y_pred: np.ndarray, save_tag: Optional[str] = None) -> None:
        """
        Plot the error per bin in a bar chart.

        :param y_true: Ground truth labels.
        :param y_pred: Predicted labels from the model.
        """
        if self.bins is None or self.min_bin_y is None or self.max_bin_y is None:
            print("Please generate bins first using get_bins()")
            return

        # Bin the true and predicted y-values
        true_bins = np.digitize(y_true, self.bins)
        pred_bins = np.digitize(y_pred, self.bins)

        # Initialize an array to store the mean errors for each bin
        bin_errors = np.zeros(len(self.bins) - 1)

        # Calculate the mean error for each bin
        for i in range(1, len(self.bins)):
            indices = np.where(true_bins == i)
            if len(indices[0]) == 0:
                continue
            true_subset = y_true[indices]
            pred_subset = y_pred[indices]
            mae = tf.keras.losses.MeanAbsoluteError()(true_subset, pred_subset).numpy()
            bin_errors[i - 1] = mae

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(bin_errors)), bin_errors,
                tick_label=[f"{self.bins[i]:.2f}-{self.bins[i + 1]:.2f}" for i in range(len(self.bins) - 1)])
        plt.xlabel('Bin Range')
        plt.ylabel('Mean Absolute Error')
        plt.title('Error per Bin')
        plt.xticks(rotation=45)
        file_path = f"test_bin_error_plot_{str(save_tag)}.png"
        plt.savefig(file_path)
        plt.close()

    def get_bins(self, y: List[float], res: float = .1) -> Tuple[List[float], float, float]:
        """
        Find the minimum and maximum values in the dataset and create a list of integer groups between those values.

        :param y: The target values in the dataset.
        :param res: The resolution of the bins.
        :return: A list of integer groups between the minimum and maximum values
        :return: The minimum bin value in the dataset.
        :return: The maximum bin value in the dataset.
        """

        # Check if y is empty, if so, return empty bins and None for min and max
        if not y:
            self.bins = []
            self.min_bin_y = None
            self.max_bin_y = None
            return self.bins, self.min_bin_y, self.max_bin_y

        # Determine the minimum and maximum values in the dataset
        self.min_bin_y = min(y)
        self.max_bin_y = max(y)

        # Calculate the number of bins based on the resolution
        self.num_bins = math.ceil((self.max_bin_y - self.min_bin_y) / res)

        # Generate the bins
        self.bins = [self.min_bin_y + i * res for i in range(self.num_bins + 1)]

        return self.bins, self.min_bin_y, self.max_bin_y

    def plot_feature_space(self, model: Model, data: ndarray, labels: ndarray, withhead: bool = False,
                           groups: Optional[List[int]] = None):
        """
        Plot the instances in the 2D feature space of the representation layer (Z).

        :param model: Trained neural network model.
        :param data: Input cme_files, shape of [num_instances, num_features].
        :param labels: Corresponding labels for the cme_files.
        :param groups: Optional grouping for labels to define color assignment.
        """

        if withhead:
            repr_layer_model = Model(inputs=model.input, outputs=model.output[0])
        else:
            # Extract the representation layer from the model
            repr_layer_model = Model(inputs=model.input, outputs=model.output)

        # Predict the representation for the input cme_files
        repr_space = repr_layer_model.predict(data)

        # If groups are defined, categorize labels into groups
        if groups:
            group_labels = [np.digitize(label, groups) for label in labels]
        else:
            group_labels = labels

        # Create a scatter plot for the representation space
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(repr_space[:, 0], repr_space[:, 1], c=group_labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('2D Feature Space of Representation Layer')
        plt.xlim(-1.5, 1.5)  # Limit x-axis to range of -1 to 1
        plt.ylim(-1.5, 1.5)  # Limit y-axis to range of -1 to 1
        plt.show()

    def check_norms(self, model: Model, X_test: np.ndarray) -> None:
        """
        Plot the L2-norms of the representation layer (Z) for the test cme_files.
        :param model: Trained neural network model.
        :param X_test:
        :return:
        """
        reprs = model.predict(X_test)
        # Calculate the norms of the representation vectors
        norms = np.linalg.norm(reprs, axis=1)
        # Find all unique norm values
        unique_norms = np.unique(norms)
        # Print or otherwise use the unique norm values
        print(unique_norms)
        # Plot a histogram of the norms and get the patches to annotate
        n, bins, patches = plt.hist(norms)

        # Iterate through the patches to annotate each bar
        for i in range(len(patches)):
            plt.text(patches[i].get_x() + patches[i].get_width() / 2,
                     patches[i].get_height(),
                     str(int(n[i])),
                     ha='center',
                     va='bottom')
        plt.title('Distribution of Norms of Representation Vectors')
        plt.xlabel('Norm Value')
        plt.ylabel('Frequency')
        plt.show()

    def count_samples_in_bins(self, y_values: ndarray):
        """
        Count the number of samples whose target values are in bins.
        The bins are defined by intervals of 0.1 from 0 to 2.

        :param y_values: A numpy array containing target values.
        :return: None, but prints the count per bin along with the range of the bin.
        """
        # Define the bins
        bins = np.arange(0, 2.1, 0.1)

        # Count the samples in each bin
        hist, _ = np.histogram(y_values, bins)

        # Output the count and range for each bin
        for i in range(len(bins) - 1):
            print(f"Bin {i} ({bins[i]:.1f} to {bins[i + 1]:.1f}): {hist[i]} samples")
