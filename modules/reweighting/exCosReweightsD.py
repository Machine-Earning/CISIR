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


def map_labels_to_reweights(labels: np.ndarray, reweights: np.ndarray) -> dict:
    """
    Map labels to their corresponding reweight values.

    Parameters:
    - labels (np.ndarray): An array of labels.
    - reweights (np.ndarray): An array of reweighting factors corresponding to each label.

    Returns:
    - dict: A dictionary where each key is a label and its value is the corresponding reweight factor.
    """
    if len(labels) != len(reweights):
        raise ValueError("Labels and reweights must be of the same length.")

    label_to_reweight_mapping = {}
    for label, reweight in zip(labels, reweights):
        label_to_reweight_mapping[float(label)] = float(reweight)

    print(f"length of labels: {len(labels)}")
    print(f"length of reweights: {len(reweights)}")
    print(f'length of label_to_reweight_mapping: {len(label_to_reweight_mapping)}')

    return label_to_reweight_mapping


class exDenseReweightsD:
    """
    Class for generating synthetic regression datasets with cosine-based reweighting.
    """
    # class variables
    debug = False
    X_train = None
    y_train = None
    min_y = None
    max_y = None
    kde = None
    min_pdf = None
    max_pdf = None
    avg_reweight = None
    reweights = None
    alpha = None
    pdf_values = None

    def __init__(self, X, y,
                 alpha: float = .9,
                 bw: [float, str] = .9,
                 min_norm_weight: Optional[float] = None,
                 tag: Optional[str] = None,
                 debug: bool = False,
                 epsilon: float = 1e-4) -> None:
        """
        Create a synthetic regression dataset with cosine-based reweighting.
        The reweighting maps the density to [0, pi/2] and applies cosine.

        :param X: Training instances.
        :param y: Training labels.
        :param bw: bandwidth for the KDE.
        :param alpha: reweighing coefficient for cosine power
        :param min_norm_weight: minimum normalized weight
        :param debug: whether to activate debugging logs
        :param epsilon: small value added to max_pdf for normalization
        """

        self.yb = None
        self.ya = None
        self.debug = debug
        self.alpha = alpha
        self.min_norm_weight = min_norm_weight

        # Create training data
        self.X_train = X
        self.y_train = y

        self.min_y = np.min(self.y_train)
        self.max_y = np.max(self.y_train)

        self.kde = gaussian_kde(self.y_train, bw_method=bw)
        # Calculate PDF values once
        self.pdf_values = self.kde.evaluate(self.y_train)
        self.min_pdf = np.min(self.pdf_values)
        self.max_pdf = np.max(self.pdf_values)
        
        # Normalize densities to [0, pi/2] range
        normalized_densities = ((self.pdf_values - self.min_pdf) / (self.max_pdf + epsilon - self.min_pdf)) * (np.pi/2)
        
        # Calculate reweighting factors using cosine
        self.reweight_factors = np.power(np.cos(normalized_densities), self.alpha)
        
        # Print debug info
        print(f"y_train min: {np.min(self.y_train)}, max: {np.max(self.y_train)}, shape: {self.y_train.shape}")
        print(f"reweight_factors shape: {self.reweight_factors.shape}")
        self.avg_reweight = np.mean(self.reweight_factors)
        self.reweights = self.reweight_factors / self.avg_reweight

        self.label_reweight_dict = map_labels_to_reweights(self.y_train, self.reweights)

        if self.debug:
            print('X_train: ', self.X_train[:12])
            print('y_train: ', self.y_train[:12])
            print('reweights: ', self.reweights[:12])
            self.plot_density_kde_reweights(tag)

    def pdf(self, y: ndarray) -> ndarray:
        """
        Probability Density Function for label y.
        :param y: The y value as a NumPy array.
        :return: The probability density at y as a NumPy array.
        """
        return self.kde.evaluate(y)

    def find_min_max_pdf(self, y: ndarray) -> None:
        """
        Find the minimum and maximum PDF values for a given NumPy array of labels y.

        :param y: A NumPy array containing labels.
        :return: None. Updates self.min_pdf and self.max_pdf.
        """
        pdf_values = self.pdf(y)
        self.min_pdf = np.min(pdf_values)
        self.max_pdf = np.max(pdf_values)

    def reweight(self, y: ndarray, alpha: float, epsilon: float = 1e-6) -> ndarray:
        """
        Calculate the reweighting factor for a label y using cosine reweighting.

        :param y: The y-value of the data points as a NumPy array.
        :param alpha: Power parameter for cosine.
        :param epsilon: Small value added to max_pdf for normalization.
        :return: The reweighting factor for the label as a NumPy array.
        """
        # Compute the density of y
        density = self.pdf(y)
        # Normalize density to [0, pi/2]
        normalized_density = (density - self.min_pdf) / (self.max_pdf + epsilon - self.min_pdf) * (np.pi/2)
        # Apply cosine reweighting
        return np.power(np.cos(normalized_density), alpha)

    def find_avg_reweight(self, y: ndarray, alpha: float):
        """
        Find the average reweighting factor for y
        :param y: labels.
        :param alpha: Power parameter for cosine.
        :return: The average reweighting factor.
        """
        reweight_factors = self.reweight(y, alpha)
        self.avg_reweight = np.mean(reweight_factors)

    def normalized_reweight(self, y: ndarray, alpha: float) -> ndarray:
        """
        Calculate the normalized reweighting factor for a label y.

        :param y: The y-value as a NumPy array.
        :param alpha: Power parameter for cosine.

        :return: The normalized reweighting factor for the label as a NumPy array.
        """
        # Ensure average reweight is not zero to avoid division by zero
        if self.avg_reweight == 0:
            raise ValueError("Average reweighting factor should not be zero.")

        reweight_factor = self.reweight(y, alpha)
        return reweight_factor / self.avg_reweight

    def preprocess_reweighting(self, y: ndarray) -> ndarray:
        """
        Returns the pre-calculated normalized reweighting factors.

        :param y: The target dataset as a NumPy array.
        :return: The normalized reweighting factors as a NumPy array.
        """
        return self.reweights
