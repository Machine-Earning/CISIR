##############################################################################################################
# Description: generating datasets for ML based on csv files you received
# (features might be added/removed over time, training/validation/test splits)
##############################################################################################################

# types for type hinting
from typing import Tuple, List, Union, Optional

import matplotlib.pyplot as plt
import mlflow
# imports
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from numpy import ndarray
from scipy.stats import gaussian_kde


class DenseJointReweights:
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
    alpha = None
    min_jpdf = None
    max_jpdf = None
    avg_jreweight = None
    jreweights = None
    jindices = None

    def __init__(self,
                 X, y,
                 alpha: float = .9,
                 bw: [float, str] = .9,
                 min_norm_weight: Optional[float] = None,
                 debug: bool = False) -> None:
        """
        Create a synthetic regression dataset.
        The input features (X) are randomly generated using a normal distribution centered at 0 with a standard
        deviation of 1.

        The target values (y) are calculated as the L2 norm (Euclidean norm) of the input features.

        :param n_train: Number of training instances.
        :param n_test: Number of testing instances.
        :param n_features: Number of input features.
        :param alpha: reweighing coefficient
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
        self.jreweights, self.jindices = self.preprocess_jreweighting(self.y_train)  # for pairs of labels

        if self.debug:
            print('X_train: ', self.X_train[:12])
            print('y_train: ', self.y_train[:12])
            print('joint indices', self.jindices[:12])
            print('joint reweights: ', self.jreweights[:12])
            self.plot_density_kde_jreweights()

    def adjust_bandwidth(self, kde: gaussian_kde, factor: Union[float, int]) -> None:
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

    def preprocess_jreweighting(self, y: ndarray) -> Tuple[ndarray, List[Tuple[int, int]]]:
        """
        Preprocess reweighting for joint PDF based on a single dataset y and
        stores the unique pairs of ya and yb labels in self.ya and self.yb.

        :param y: The target dataset as a NumPy array.
        :return: A tuple where the first element is the normalized_joint_factors and the second
                 element is a list of tuple pairs that correspond to the indices in y making up self.ya and self.yb.
        """

        # Step 1: Find all unique pairs of ya and yb
        n = len(y)
        i, j = np.triu_indices(n, k=1)  # Get upper triangle indices, excluding diagonal
        self.ya, self.yb = y[i], y[j]  # Get the unique pairs of ya and yb

        # Convert to NumPy arrays for efficient numerical operations
        self.ya = np.array(self.ya)
        self.yb = np.array(self.yb)

        # Step 2: Find min and max joint PDF values and store them
        self.find_min_max_jpdf(self.ya, self.yb)

        # Step 3: Find average joint reweighting factor
        self.find_avg_jreweight(self.ya, self.yb, self.alpha)

        # Step 4: Calculate normalized joint reweighting factors
        normalized_joint_factors = self.normalized_jreweight(self.ya, self.yb, self.alpha)
        # Create a list of index pairs corresponding to ya and yb
        index_pairs = list(zip(i, j))

        return normalized_joint_factors, index_pairs

    def normalized_jreweight(self, ya: ndarray, yb: ndarray, alpha: float, epsilon: float = 1e-7) -> ndarray:
        """
        Calculate the normalized reweighting factor for joint labels ya and yb.

        :param ya, yb: The y-values as NumPy arrays.
        :param alpha: Parameter to adjust the reweighting.
        :param epsilon: A small constant to avoid zero reweighting.
        :return: The normalized reweighting factor for the labels as a NumPy array.
        """
        # Ensure average reweight is not zero to avoid division by zero
        if self.avg_jreweight == 0:
            raise ValueError("Average reweighting factor should not be zero.")

        jreweight_factor = self.jreweight(ya, yb, alpha, epsilon)
        normalized_joint_factor = jreweight_factor / self.avg_jreweight

        return normalized_joint_factor

    def find_avg_jreweight(self, ya: ndarray, yb: ndarray, alpha: float, epsilon: float = 1e-7) -> float:
        """
        Find the average reweighting factor for joint labels ya and yb.
        :param ya, yb: labels.
        :param alpha: Parameter to adjust the reweighting.
        :param epsilon: A small constant to avoid zero reweighting.
        :return: The average reweighting factor.
        """

        total_jreweight = np.sum(self.jreweight(ya, yb, alpha, epsilon))
        count = len(ya)

        self.avg_jreweight = total_jreweight / count if count > 0 else 0

    def jreweight(self, ya: ndarray, yb: ndarray, alpha: float, epsilon: float = 1e-7) -> ndarray:
        """
        Calculate the reweighting factor for joint labels ya and yb.

        :param ya, yb: The y-values of the cme_files points as NumPy arrays.
        :param alpha: Parameter to adjust the reweighting.
        :param epsilon: A small constant to avoid zero reweighting.
        :return: The reweighting factor for the labels as a NumPy array.
        """
        # Compute the joint density
        joint_density = self.jpdf(ya, yb)

        # Normalize the joint density
        normalized_jpdf = (joint_density - self.min_jpdf) / (self.max_jpdf - self.min_jpdf)

        # Compute the reweighting factor
        if self.min_norm_weight is not None:
            epsilon = self.min_norm_weight ** 2
        jreweighting_factor = np.maximum(1 - alpha * normalized_jpdf, epsilon)

        return jreweighting_factor

    def find_min_max_jpdf(self, ya: ndarray, yb: ndarray) -> None:
        """
        Find the minimum and maximum joint PDF values for a given NumPy array of labels ya and yb.

        :param ya: A NumPy array containing labels for the first variable.
        :param yb: A NumPy array containing labels for the second variable.
        :return: None. Updates self.min_jpdf and self.max_jpdf.
        """
        joint_pdf_values = self.jpdf(ya, yb)
        # NOTE: this assume ya and yb are augmented with
        # all the possible pairings of original ya and yb
        # and this ya and yb will admit duplicates

        self.min_jpdf = np.min(joint_pdf_values)
        self.max_jpdf = np.max(joint_pdf_values)

    def compute_joint_kde(self, ya_values, yb_values):
        """
        Computes the joint KDE for a set of ya and yb values.
        """
        # Assume self.kde is already a gaussian_kde object fitted with y_train cme_files
        kde_ya = self.kde.evaluate(ya_values)
        kde_yb = self.kde.evaluate(yb_values)
        return kde_ya * kde_yb

    def jpdf(self, ya: ndarray, yb: ndarray) -> ndarray:
        """
        Joint Probability Density Function for labels ya and yb.

        :param ya: The y value for the first variable as a NumPy array.
        :param yb: The y value for the second variable as a NumPy array.
        :return: The joint probability density as a NumPy array.
        """
        return self.kde.evaluate(ya) * self.kde.evaluate(yb)

    def compute_joint_hist_density(self, ya_values, yb_values, bins=30):
        """
        Computes the joint histogram density for a set of ya and yb values.

        :param ya_values: NumPy array of ya values to be binned.
        :param yb_values: NumPy array of yb values to be binned.
        :param bins: Number of bins or a sequence defining the bin edges.
        :return: Joint histogram density as a 2D NumPy array.
        """
        hist, x_edges, y_edges = np.histogram2d(ya_values, yb_values, bins=bins, density=True)

        # Create 2D array representing the bin centers
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)

        return hist, X, Y

    def plot_density_kde_jreweights(self):
        """
        Plot the joint label density, joint KDE, and joint reweights as separate subplots.
        """
        y_values = np.linspace(self.min_y, self.max_y, 100)
        Y1, Y2 = np.meshgrid(y_values, y_values)

        ya_values = Y1.ravel()
        yb_values = Y2.ravel()

        joint_hist_density, X, Y = self.compute_joint_hist_density(ya_values, yb_values)
        joint_kde_values = self.compute_joint_kde(ya_values, yb_values)
        joint_reweights = self.normalized_jreweight(ya_values, yb_values, self.alpha)

        joint_kde_values = joint_kde_values.reshape(Y1.shape)
        joint_reweights = joint_reweights.reshape(Y1.shape)

        fig = plt.figure(figsize=(18, 18))

        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(X, Y, joint_hist_density, cmap='viridis', alpha=0.7)
        ax1.set_title('Joint Density')
        ax1.set_xlabel('ya')
        ax1.set_ylabel('yb')
        ax1.set_zlabel('Value')

        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_surface(Y1, Y2, joint_kde_values, cmap='coolwarm', alpha=0.7)
        ax2.set_title('Joint KDE')
        ax2.set_xlabel('ya')
        ax2.set_ylabel('yb')
        ax2.set_zlabel('Value')

        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot_surface(Y1, Y2, joint_reweights, cmap='autumn', alpha=0.7)
        ax3.set_title('Joint Reweights')
        ax3.set_xlabel('ya')
        ax3.set_ylabel('yb')
        ax3.set_zlabel('Value')

        plt.show()


class DenseReweights:
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

        :param n_train: Number of training instances.
        :param n_test: Number of testing instances.
        :param n_features: Number of input features.
        :param alpha: rewweighing coefficient
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

        if self.debug:
            print('X_train: ', self.X_train[:12])
            print('y_train: ', self.y_train[:12])
            print('reweights: ', self.reweights[:12])
            self.plot_density_kde_reweights(tag)

    def adjust_bandwidth(self, kde: gaussian_kde, factor: Union[float, int]) -> None:
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

    def plot_distributions(self, y_train, y_val):
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

    def get_density_at_points(self, kde, points):
        """
        Get the density of a KDE model at specific points.

        Parameters:
        - kde (scipy.stats.gaussian_kde): The KDE model.
        - points (List[float]): List of points where the density needs to be evaluated.

        Returns:
        - List[float]: List of densities at the given points.
        """
        return kde.evaluate(points)

    def plot_density_kde_reweights(self, tag: Optional[str] = None):
        """
        Plot the label density, KDE, and reweights for the y_train dataset.
        """
        # Points where you want to find the density
        points_to_evaluate = [self.min_y, self.max_y, self.max_y - 1, 1.4]
        # Get the density at these points
        density_values = self.get_density_at_points(self.kde, points_to_evaluate)
        # Print the density values
        print(f"Density at min_y background ({self.min_y}): {density_values[0]}")
        print(f"Density at max_y Sep ({self.max_y}): {density_values[1]}")
        print(f"Density at max_y lower Sep ({self.max_y - 1}): {density_values[2]}")
        print(f"Density at y=1.4 elevated: {density_values[3]}")
        # print the probability of background, elevated, and seps
        event_probs = self.calc_event_probs(self.y_train, self.kde)
        print(f'event probabilities: {event_probs}')
        kde_ratio = event_probs["background"] / event_probs["sep"]
        print(f'KDE background to SEP ratio: {kde_ratio}')
        # get background to sep ratio in frequency
        background_threshold: float = np.log(10 / np.exp(2))
        sep_threshold: float = np.log(10)
        background_count = np.sum(self.y_train <= background_threshold)
        sep_count = np.sum(self.y_train > sep_threshold)
        # Avoid division by zero
        if sep_count == 0:
            return float('inf')  # Return infinity if there are no SEP events
        freq_ratio = background_count / sep_count
        print(f'Frequency background to SEP ratio: {freq_ratio}')
        # with mlflow.start_run():
        # Log parameters like min_y and max_y
        mlflow.log_param("min_y", self.min_y)
        mlflow.log_param("max_y", self.max_y)
        # Print and Log density values
        mlflow.log_metric("density_min_y_background", density_values[0])
        mlflow.log_metric("density_max_y_sep", density_values[1])
        mlflow.log_metric("density_lower_sep", density_values[2])
        mlflow.log_metric("density_elevated", density_values[3])
        mlflow.log_metric("prob_background", event_probs["background"])
        mlflow.log_metric("prob_sep", event_probs["sep"])
        mlflow.log_metric("prob_elevated", event_probs["elevated"])
        mlflow.log_metric("kde_background_to_sep_ratio", kde_ratio)
        mlflow.log_metric("freq_background_to_sep_ratio", freq_ratio)

        # Compute KDE values at the sample y values
        kde_values_samples = self.kde.evaluate(self.y_train)
        # Get reweights for the sample y values
        reweights_samples = self.normalized_reweight(self.y_train, self.alpha)

        fig, axs = plt.subplots(4, 1, figsize=(12, 16))

        # Formatter for two decimal places
        formatter = FormatStrFormatter('%.2f')

        # Filter for positive ln(Intensity)
        positive_mask = self.y_train > 0
        y_train_positive = self.y_train[positive_mask]
        kde_values_samples_positive = kde_values_samples[positive_mask]
        reweights_samples_positive = reweights_samples[positive_mask]

        # Plot for KDE with all y values
        axs[0].grid(True, which="both", ls="--", c='#dddddd', zorder=0)
        axs[0].scatter(self.y_train, kde_values_samples, label='KDE', color='green', alpha=0.7, s=10, zorder=5)
        axs[0].xaxis.set_major_formatter(formatter)
        axs[0].set_xticks(np.linspace(min(self.y_train), max(self.y_train), 10))
        axs[0].set_xlabel('ln(Intensity)')
        axs[0].set_ylabel('KDE')
        axs[0].legend()
        axs[0].set_title(f'KDE for all y values, kde factor {self.kde.factor}')

        # Plot for KDE with positive ln(Intensity)
        axs[1].grid(True, which="both", ls="--", c='#dddddd', zorder=0)
        axs[1].scatter(y_train_positive, kde_values_samples_positive, label='KDE', color='green', alpha=0.7, s=10,
                       zorder=5)
        axs[1].xaxis.set_major_formatter(formatter)
        axs[1].set_xticks(np.linspace(min(y_train_positive), max(y_train_positive), 10))
        axs[1].set_xlabel('ln(Intensity)')
        axs[1].set_ylabel('KDE')
        axs[1].legend()
        axs[1].set_title('KDE for positive ln(Intensity)')

        # Plot for Reweights with all y values
        axs[2].grid(True, which="both", ls="--", c='#dddddd', zorder=0)
        axs[2].scatter(self.y_train, reweights_samples, label='Reweights', color='red', alpha=0.7, s=10, zorder=5)
        axs[2].xaxis.set_major_formatter(formatter)
        axs[2].set_xticks(np.linspace(min(self.y_train), max(self.y_train), 10))
        axs[2].set_xlabel('ln(Intensity)')
        axs[2].set_ylabel('Reweights')
        axs[2].legend()
        axs[2].set_title('Reweights for all y values')

        # Plot for Reweights with positive ln(Intensity)
        axs[3].grid(True, which="both", ls="--", c='#dddddd', zorder=0)
        axs[3].scatter(y_train_positive, reweights_samples_positive, label='Reweights', color='red', alpha=0.7, s=10,
                       zorder=5)
        axs[3].xaxis.set_major_formatter(formatter)
        axs[3].set_xticks(np.linspace(min(y_train_positive), max(y_train_positive), 10))
        axs[3].set_xlabel('ln(Intensity)')
        axs[3].set_ylabel('Reweights')
        axs[3].legend()
        axs[3].set_title('Reweights for positive ln(Intensity)')

        plt.tight_layout()
        plt.savefig(tag)
        plt.close()

    def calc_event_probs(self, y_values: np.ndarray, kde: gaussian_kde, decimal_places: int = 4) -> dict:
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

    def reweight(self, y: ndarray, alpha: float, epsilon: float = 1e-7) -> ndarray:
        """
        Calculate the reweighting factor for a label y.

        :param y: The y-value of the cme_files points as a NumPy array.
        :param alpha: Parameter to adjust the reweighting.
        :param epsilon: A small constant to avoid zero reweighting.
        :return: The reweighting factor for the label as a NumPy array.
        """
        # Compute the density of y
        density = self.pdf(y)

        # Normalize the joint density
        normalized_pdf = (density - self.min_pdf) / (self.max_pdf - self.min_pdf)

        # Compute the reweighting factor
        if self.min_norm_weight is not None:
            epsilon = self.min_norm_weight
        reweighting_factor = np.maximum(1 - alpha * normalized_pdf, epsilon)

        return reweighting_factor

    def find_avg_reweight(self, y: ndarray, alpha: float, epsilon: float = 1e-7) -> float:
        """
        Find the average reweighting factor for y
        :param y: labels.
        :param alpha: Parameter to adjust the reweighting.
        :param epsilon: A small constant to avoid zero reweighting.
        :return: The average reweighting factor.
        """

        total_reweight = np.sum(self.reweight(y, alpha, epsilon))
        count = len(y)

        self.avg_reweight = total_reweight / count if count > 0 else 0

    def normalized_reweight(self, y: ndarray, alpha: float, epsilon: float = 1e-7) -> ndarray:
        """
        Calculate the normalized reweighting factor for a label y.

        :param y: The y-value as a NumPy array.
        :param alpha: Parameter to adjust the reweighting.
        :param epsilon: A small constant to avoid zero reweighting.
        :return: The normalized reweighting factor for the label as a NumPy array.
        """
        # Ensure average reweight is not zero to avoid division by zero
        if self.avg_reweight == 0:
            raise ValueError("Average reweighting factor should not be zero.")

        reweight_factor = self.reweight(y, alpha, epsilon)
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
