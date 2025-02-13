##############################################################################################################
# Description: generating datasets for ML based on csv files you received
# (features might be added/removed over time, training/validation/test splits)
##############################################################################################################

# types for type hinting
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.stats import gaussian_kde


def adjust_bandwidth(kde: gaussian_kde, factor: Union[float, int]) -> None:
    """
    Adjust the bandwidth of a given KDE object by a multiplicative factor.

    Parameters:
    - kde (gaussian_kde): The KDE object whose bandwidth needs to be adjusted.
    - factor (float|int): The factor by which to adjust the bandwidth.

    Returns:
    - None: The function modifies the KDE object in-place.
    """
    # Obtain the original bandwidth (factor)
    original_bw = kde.factor

    # Calculate the adjusted bandwidth
    adjusted_bw = original_bw * factor

    # Set the adjusted bandwidth back into the KDE object
    kde.set_bandwidth(bw_method=adjusted_bw)


def plot_distributions(y_train, y_val):
    """
    Plot the sorted distributions of training and validation labels.

    :param y_train: Training labels
    :param y_val: Validation labels
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Sorted Training Labels Distribution")
    plt.plot(np.sort(y_train), marker='o', linestyle='')
    plt.xlabel("Index")
    plt.ylabel("Value")

    plt.subplot(1, 2, 2)
    plt.title("Sorted Validation Labels Distribution")
    plt.plot(np.sort(y_val), marker='o', linestyle='')
    plt.xlabel("Index")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()


def get_density_at_points(kde, points):
    """
    Get the density of a KDE model at specific points.

    Parameters:
    - kde (scipy.stats.gaussian_kde): The KDE model.
    - points (List[float]): List of points where the density needs to be evaluated.

    Returns:
    - List[float]: List of densities at the given points.
    """
    return kde.evaluate(points)


def calc_event_probs(y_values: np.ndarray, kde: gaussian_kde, decimal_places: int = 4) -> dict:
    """
    Calculate the probabilities of background, elevated, and sep events based on given thresholds using KDE.

    Parameters:
    - y_values (np.ndarray): The array of y-values to check.
    - kde (gaussian_kde): The KDE object for the y_values.
    - decimal_places (int): The number of decimal places for the probabilities. Default is 4.

    Returns:
    - dict: Dictionary containing probabilities of each event type rounded to the specified number of decimal places.
    """
    background_threshold: float = np.log(10 / np.exp(2))
    sep_threshold: float = np.log(10)
    # Create a range of y values for integrating KDE
    y_range = np.linspace(min(y_values), max(y_values), 1000)

    # Evaluate the KDE across the y_range
    kde_values = kde.evaluate(y_range)

    # Calculate the integral (area under curve) using trapezoidal rule
    total_area = np.trapz(kde_values, y_range)

    # Calculate area for each event type
    background_area = np.trapz(kde_values[y_range <= background_threshold],
                               y_range[y_range <= background_threshold])
    elevated_area = np.trapz(kde_values[(y_range > background_threshold) & (y_range <= sep_threshold)],
                             y_range[(y_range > background_threshold) & (y_range <= sep_threshold)])
    sep_area = np.trapz(kde_values[y_range > sep_threshold], y_range[y_range > sep_threshold])

    # Calculate probabilities based on areas
    probabilities = {
        "background": round(background_area / total_area, decimal_places),
        "elevated": round(elevated_area / total_area, decimal_places),
        "sep": round(sep_area / total_area, decimal_places)
    }

    return probabilities


def map_labels_to_importance_weights(labels: np.ndarray, importance_weights: np.ndarray) -> dict:
    """
    Map labels to their corresponding importance weights.

    Parameters:
    - labels (np.ndarray): An array of labels.
    - importance_weights (np.ndarray): An array of importance weights corresponding to each label.

    Returns:
    - dict: A dictionary where each key is a label and its value is the corresponding importance weight.
    """
    if len(labels) != len(importance_weights):
        raise ValueError("Labels and importance weights must be of the same length.")

    label_to_importance_weight_mapping = {}
    for label, importance_weight in zip(labels, importance_weights):
        label_to_importance_weight_mapping[float(label)] = float(importance_weight)

    # print(f"length of labels: {len(labels)}")
    # print(f"length of importance_weights: {len(importance_weights)}")
    # print(f'length of label_to_importance_weight_mapping: {len(label_to_importance_weight_mapping)}')

    return label_to_importance_weight_mapping


def smooth_kde_tails(x: np.ndarray, density: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    """
    TODO: figure out how to smooth the tails of the KDE without losing the shape of the KDE
    Smooths density values within the threshold range by detecting and interpolating outliers.
    An outlier is defined as a density value that is significantly lower than its neighbors.

    Parameters:
    - x: np.ndarray
      The input points where density is evaluated.
    - density: np.ndarray
      The original KDE density values.
    - threshold: float
      The absolute value threshold defining the range to consider for outlier detection.

    Returns:
    - np.ndarray
      Density values with outliers replaced by linear interpolation of neighbors.
    """
    smoothed_density = density.copy()
    
    # Get indices within threshold range
    mask = (x <= -threshold) | (x >= threshold)
    indices = np.where(mask)[0]
    
    if len(indices) < 3:  # Need at least 3 points for outlier detection
        return smoothed_density
        
    # Calculate density differences with neighbors
    density_diffs = np.abs(density[indices[1:-1]] - np.mean([density[indices[:-2]], density[indices[2:]]], axis=0))
    
    # Define outliers as points with density < 2x average of neighbors
    outlier_indices = indices[1:-1][density_diffs < 2.0 * np.mean([density[indices[:-2]], density[indices[2:]]], axis=0)]
    
    # Replace outliers with linear interpolation of valid neighbors
    for idx in outlier_indices:
        # Find valid left neighbor (not an outlier)
        left_idx = idx - 1
        while left_idx in outlier_indices and left_idx > 0:
            left_idx -= 1
            
        # Find valid right neighbor (not an outlier)  
        right_idx = idx + 1
        while right_idx in outlier_indices and right_idx < len(x) - 1:
            right_idx += 1
            
        smoothed_density[idx] = np.interp(x[idx], [x[left_idx], x[right_idx]], 
                                        [density[left_idx], density[right_idx]])
    
    return smoothed_density

class ReciprocalImportance:
    """
    Class for generating importance weights based on the reciprocal of the PDF,
    with smoothed tails to prevent noisy importance weights at the extremes.
    """
    def __init__(
            self, 
            features: np.ndarray, 
            labels: np.ndarray,
            alpha: float = 1,
            bandwidth: Union[float, str] = 0.07,
            epsilon: float = 1e-3
        ) -> None:
        """
        Initialize the ReciprocalImportanceWeighting class.

        Parameters:
        - features: np.ndarray - Training instances
        - labels: np.ndarray - Training labels
        - bandwidth: Union[float, str] - Bandwidth for the KDE
        - alpha: float - Importance weight coefficient
        - epsilon: float - Small value added to max_pdf for normalization
        """
        self.alpha = alpha
        self.features = features
        self.labels = labels

        # Create KDE and get PDF values
        self.kde = gaussian_kde(self.labels, bw_method=bandwidth)
        self.densities = self.kde.evaluate(self.labels)
        self.min_density = np.min(self.densities)
        self.max_density = np.max(self.densities)

        # Calculate normalized PDF so PDF is between 0 and 1 using vectorized ops
        normalized_densities = np.divide(self.densities, self.max_density + epsilon)

        # # print the min and max of the normalized densities
        # print(f"min of normalized densities: {np.min(normalized_densities)}")
        # print(f"max of normalized densities: {np.max(normalized_densities)}")

        # Calculate reweighting factors using vectorized ops
        importance_weights = np.power(np.reciprocal(normalized_densities + 1e-8), alpha)

        # Calculate average importance weight and normalize using vectorized ops
        self.avg_importance_weight = np.mean(importance_weights)
        self.importance_weights = np.divide(importance_weights, self.avg_importance_weight)

        # Create mapping dictionary
        self.label_importance_map = map_labels_to_importance_weights(
            self.labels, self.importance_weights)


class QUCImportance:
    """
    Class for generating importance weights based on the Quarter of the Unit Circle (QUC).
    """

    def __init__(
            self, 
            features: np.ndarray, 
            labels: np.ndarray,
            alpha: float = 2,
            bandwidth: Union[float, str] = .07,
            epsilon: float = 1e-3
        ) -> None:
        """
        Initialize the QUCImportance class.

        The reweighting uses importance_weight = [1- density^(alpha)]^(1/alpha) where density is normalized density between 0 and 1.

        :param features: Training instances.
        :param labels: Training labels.
        :param bandwidth: bandwidth for the KDE.
        :param alpha: importance weight coefficient for power function
        :param epsilon: small value added to max_pdf for normalization
        """

        # Create training data
        self.features = features
        self.labels = labels

        # Create KDE and get PDF values
        self.kde = gaussian_kde(self.labels, bw_method=bandwidth)
        self.densities = self.kde.evaluate(self.labels)
        self.min_density = np.min(self.densities)
        self.max_density = np.max(self.densities)
        
        # Normalize densities to [0, 1] range with max_pdf + epsilon mapping to 1
        normalized_densities = np.divide(self.densities, self.max_density + epsilon)
        
        # Calculate reweighting factors using power function
        # y = [1- x^(alpha)]^(1/alpha) where x is normalized density
        self.reweight_factors = np.power(1 - np.power(normalized_densities, alpha), 1.0 / alpha)
        
        # Calculate average importance weight and normalize using vectorized ops
        self.avg_importance_weight = np.mean(self.reweight_factors)
        self.importance_weights = np.divide(self.reweight_factors, self.avg_importance_weight)

        # Create mapping dictionary
        self.label_importance_map = map_labels_to_importance_weights(
            self.labels, self.importance_weights)
        


class LinearImportance:
    """
    Class for generating importance weights based on the linear function.
    """

    def __init__(
            self, 
            features: np.ndarray, 
            labels: np.ndarray,
            alpha: float = 1,
            bandwidth: Union[float, str] = .07,
            epsilon: float = 1e-3
        ) -> None:
        """
        Initialize the LinearImportance class.

        :param X: Training instances.
        :param y: Training labels.
        :param bandwidth: bandwidth for the KDE.
        :param alpha: reweighing coefficient
        :param epsilon: small value added to max_pdf for normalization
        """

        self.features = features
        self.labels = labels

        self.kde = gaussian_kde(self.labels, bw_method=bandwidth)
        # Calculate PDF values once
        self.densities = self.kde.evaluate(self.labels)
        self.min_density = np.min(self.densities)
        self.max_density = np.max(self.densities)

        # Normalize densities to [0, 1] range with max_pdf + epsilon mapping to 1
        normalized_densities = np.divide(self.densities, self.max_density + epsilon)
        
        # Calculate reweighting factors once, ensuring no negative values
        self.importance_weights = alpha * (self.max_density - self.densities)
        
        # Calculate average importance weight and normalize using vectorized ops
        self.avg_importance_weight = np.mean(self.importance_weights)
        self.importance_weights = np.divide(self.importance_weights, self.avg_importance_weight)

        self.label_importance_map = map_labels_to_importance_weights(
            self.labels, self.importance_weights)


class CosineImportance:
    """
    Class for generating importance weights based on the cosine function.
    """

    def __init__(
            self, 
            features: np.ndarray, 
            labels: np.ndarray,
            alpha: float = 1,
            bandwidth: Union[float, str] = .07,
            epsilon: float = 1e-3
        ) -> None:
        """
        Initialize the CosineImportance class.

        The reweighting maps the density to [0, pi/2] and applies cosine.

        :param features: Training instances.
        :param labels: Training labels.
        :param bandwidth: bandwidth for the KDE.
        :param alpha: reweighing coefficient for cosine power
        :param epsilon: small value added to max_pdf for normalization
        """

        self.features = features
        self.labels = labels

        # Create KDE and get PDF values
        self.kde = gaussian_kde(self.labels, bw_method=bandwidth)
        # Calculate PDF values once
        self.densities = self.kde.evaluate(self.labels)
        self.min_density = np.min(self.densities)
        self.max_density = np.max(self.densities)
        
        # Normalize densities to [0, pi/2] range using vectorized operations
        normalized_densities = np.multiply(
            np.divide(
                np.subtract(self.densities, self.min_density),
                np.subtract(
                    np.add(self.max_density, epsilon),
                    self.min_density
                )
            ),
            np.pi / 2.0
        )
        
        # Calculate importance weights using cosine
        self.importance_weights = np.power(np.cos(normalized_densities), alpha)
        
        # Calculate average importance weight and normalize using vectorized ops
        self.avg_importance_weight = np.mean(self.importance_weights)
        self.importance_weights = np.divide(self.importance_weights, self.avg_importance_weight)

        self.label_importance_map = map_labels_to_importance_weights(
            self.labels, self.importance_weights)

