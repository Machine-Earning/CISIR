##############################################################################################################
# Description: generating datasets for ML based on csv files you received
# (features might be added/removed over time, training/validation/test splits)
##############################################################################################################

# types for type hinting
from typing import Tuple, List, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
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
    Class for generating synthetic regression datasets.
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

    def __init__(self, X, y,
                 alpha: float = .9,
                 bw: [float, str] = .9,
                 min_norm_weight: Optional[float] = None,
                 tag: Optional[str] = None,
                 debug: bool = False) -> None:
        """
        Create a synthetic regression dataset.
        The input features (X) are randomly generated using a normal distribution centered at 0 with a standard
        deviation of 1.

        The target values (y) are calculated as the L2 norm (Euclidean norm) of the input features.

        :param X: Training instances.
        :param y: Training labels.
        :param bw: bandwidth for the KDE.
        :param alpha: reweighing coefficient
        :param min_norm_weight: minimum normalized weight
        :param debug: whether to activate debugging logs
        """

        self.yb = None
        self.ya = None
        self.debug = debug
        self.alpha = alpha
        self.min_norm_weight = min_norm_weight

        # Create training cme_files
        self.X_train = X
        self.y_train = y

        self.min_y = np.min(self.y_train)
        self.max_y = np.max(self.y_train)

        self.kde = gaussian_kde(self.y_train, bw_method=bw)
        # self.adjust_bandwidth(self.kde, bw_factor)
        self.reweights = self.preprocess_reweighting(self.y_train)  # for labels, order maintained

        self.label_reweight_dict = map_labels_to_reweights(self.y_train, self.reweights)

        if self.debug:
            print('X_train: ', self.X_train[:12])
            print('y_train: ', self.y_train[:12])
            print('reweights: ', self.reweights[:12])
            self.plot_density_kde_reweights(tag)

    # def plot_density_kde_reweights(self, tag: Optional[str] = None):
    #     """
    #     Plot the label density, KDE, and reweights for the y_train dataset.
    #     """
    #     # Points where you want to find the density
    #     points_to_evaluate = [self.min_y, self.max_y, self.max_y - 1, 1.4]
    #     # Get the density at these points
    #     density_values = get_density_at_points(self.kde, points_to_evaluate)
    #     # Print the density values
    #     print(f"Density at min_y background ({self.min_y}): {density_values[0]}")
    #     print(f"Density at max_y Sep ({self.max_y}): {density_values[1]}")
    #     print(f"Density at max_y lower Sep ({self.max_y - 1}): {density_values[2]}")
    #     print(f"Density at y=1.4 elevated: {density_values[3]}")
    #     # print the probability of background, elevated, and seps
    #     event_probs = calc_event_probs(self.y_train, self.kde)
    #     print(f'event probabilities: {event_probs}')
    #     kde_ratio = event_probs["background"] / event_probs["sep"]
    #     print(f'KDE background to SEP ratio: {kde_ratio}')
    #     # get background to sep ratio in frequency
    #     background_threshold: float = np.log(10 / np.exp(2))
    #     sep_threshold: float = np.log(10)
    #     background_count = np.sum(self.y_train <= background_threshold)
    #     sep_count = np.sum(self.y_train > sep_threshold)
    #     # Avoid division by zero
    #     if sep_count == 0:
    #         return float('inf')  # Return infinity if there are no SEP events
    #     freq_ratio = background_count / sep_count
    #     print(f'Frequency background to SEP ratio: {freq_ratio}')
    #
    #     # Compute KDE values at the sample y values
    #     kde_values_samples = self.kde.evaluate(self.y_train)
    #     # Get reweights for the sample y values
    #     reweights_samples = self.normalized_reweight(self.y_train, self.alpha)
    #
    #     fig, axs = plt.subplots(4, 1, figsize=(12, 16))
    #
    #     # Formatter for two decimal places
    #     formatter = FormatStrFormatter('%.2f')
    #
    #     # Filter for positive ln(Intensity)
    #     positive_mask = self.y_train > 0
    #     y_train_positive = self.y_train[positive_mask]
    #     kde_values_samples_positive = kde_values_samples[positive_mask]
    #     reweights_samples_positive = reweights_samples[positive_mask]
    #
    #     # Plot for KDE with all y values
    #     axs[0].grid(True, which="both", ls="--", c='#dddddd', zorder=0)
    #     axs[0].scatter(self.y_train, kde_values_samples, label='KDE', color='green', alpha=0.7, s=10, zorder=5)
    #     axs[0].xaxis.set_major_formatter(formatter)
    #     axs[0].set_xticks(np.linspace(min(self.y_train), max(self.y_train), 10))
    #     axs[0].set_xlabel('ln(Intensity)')
    #     axs[0].set_ylabel('KDE')
    #     axs[0].legend()
    #     axs[0].set_title(f'KDE for all y values, kde factor {self.kde.factor}')
    #
    #     # Plot for KDE with positive ln(Intensity)
    #     axs[1].grid(True, which="both", ls="--", c='#dddddd', zorder=0)
    #     axs[1].scatter(y_train_positive, kde_values_samples_positive, label='KDE', color='green', alpha=0.7, s=10,
    #                    zorder=5)
    #     axs[1].xaxis.set_major_formatter(formatter)
    #     axs[1].set_xticks(np.linspace(min(y_train_positive), max(y_train_positive), 10))
    #     axs[1].set_xlabel('ln(Intensity)')
    #     axs[1].set_ylabel('KDE')
    #     axs[1].legend()
    #     axs[1].set_title('KDE for positive ln(Intensity)')
    #
    #     # Plot for Reweights with all y values
    #     axs[2].grid(True, which="both", ls="--", c='#dddddd', zorder=0)
    #     axs[2].scatter(self.y_train, reweights_samples, label='Reweights', color='red', alpha=0.7, s=10, zorder=5)
    #     axs[2].xaxis.set_major_formatter(formatter)
    #     axs[2].set_xticks(np.linspace(min(self.y_train), max(self.y_train), 10))
    #     axs[2].set_xlabel('ln(Intensity)')
    #     axs[2].set_ylabel('Reweights')
    #     axs[2].legend()
    #     axs[2].set_title('Reweights for all y values')
    #
    #     # Plot for Reweights with positive ln(Intensity)
    #     axs[3].grid(True, which="both", ls="--", c='#dddddd', zorder=0)
    #     axs[3].scatter(y_train_positive, reweights_samples_positive, label='Reweights', color='red', alpha=0.7,
    #                    s=10,
    #                    zorder=5)
    #     axs[3].xaxis.set_major_formatter(formatter)
    #     axs[3].set_xticks(np.linspace(min(y_train_positive), max(y_train_positive), 10))
    #     axs[3].set_xlabel('ln(Intensity)')
    #     axs[3].set_ylabel('Reweights')
    #     axs[3].legend()
    #     axs[3].set_title('Reweights for positive ln(Intensity)')
    #
    #     plt.tight_layout()
    #     plt.savefig(tag)
    #     plt.close()

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

    def reweight(self, y: ndarray, alpha: float) -> ndarray:
        """
        Calculate the reweighting factor for a label y.

        :param y: The y-value of the cme_files points as a NumPy array.
        :param alpha: Parameter to adjust the reweighting.
        :return: The reweighting factor for the label as a NumPy array.
        """
        # Compute the density of y
        density = self.pdf(y)

        # Normalize the joint density
        normalized_pdf = (density - self.min_pdf) / (self.max_pdf - self.min_pdf)

        # Compute the reweighting factor
        # if self.min_norm_weight is not None:
        #     epsilon = self.min_norm_weight
        reweighting_factor = (1 / (normalized_pdf + 1e-8)) ** alpha
        # the 1e-8 is to avoid division by zero

        return reweighting_factor

    def find_avg_reweight(self, y: ndarray, alpha: float):
        """
        Find the average reweighting factor for y
        :param y: labels.
        :param alpha: Parameter to adjust the reweighting.
        :return: The average reweighting factor.
        """

        total_reweight = np.sum(self.reweight(y, alpha))
        count = len(y)

        self.avg_reweight = total_reweight / count if count > 0 else 0

    def normalized_reweight(self, y: ndarray, alpha: float) -> ndarray:
        """
        Calculate the normalized reweighting factor for a label y.

        :param y: The y-value as a NumPy array.
        :param alpha: Parameter to adjust the reweighting.

        :return: The normalized reweighting factor for the label as a NumPy array.
        """
        # Ensure average reweight is not zero to avoid division by zero
        if self.avg_reweight == 0:
            raise ValueError("Average reweighting factor should not be zero.")

        reweight_factor = self.reweight(y, alpha)
        normalized_factor = reweight_factor / self.avg_reweight

        return normalized_factor

    def preprocess_reweighting(self, y: ndarray) -> ndarray:
        """
        Preprocess reweighting for a dataset y and returns the normalized reweighting factors.

        :param y: The target dataset as a NumPy array.
        :return: The normalized reweighting factors as a NumPy array.
        """

        # Step 1: Find min and max pdf values and store them
        self.find_min_max_pdf(y)

        # Step 2: Find average reweighting factor
        self.find_avg_reweight(y, self.alpha)

        # Step 3: Calculate normalized reweighting factors for the dataset y
        normalized_factors = self.normalized_reweight(y, self.alpha)

        return normalized_factors
