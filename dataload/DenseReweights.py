##############################################################################################################
# Description: generating datasets for ML based on csv files you received
# (features might be added/removed over time, training/validation/test splits)
##############################################################################################################

# imports
import numpy as np
from numpy import ndarray
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
# types for type hinting
from typing import Tuple, List


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

    def __init__(self, X, y, alpha: float = .9,
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

        # Create training data
        self.X_train = X
        self.y_train = y

        self.min_y = np.min(self.y_train)
        self.max_y = np.max(self.y_train)

        self.kde = gaussian_kde(self.y_train, bw_method='scott')
        self.jreweights, self.jindices = self.preprocess_jreweighting(self.y_train)  # for pairs of labels

        if self.debug:
            print('X_train: ', self.X_train[:12])
            print('y_train: ', self.y_train[:12])
            print('joint indices', self.jindices[:12])
            print('joint reweights: ', self.jreweights[:12])
            self.plot_density_kde_jreweights()

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

        :param ya, yb: The y-values of the data points as NumPy arrays.
        :param alpha: Parameter to adjust the reweighting.
        :param epsilon: A small constant to avoid zero reweighting.
        :return: The reweighting factor for the labels as a NumPy array.
        """
        # Compute the joint density
        joint_density = self.jpdf(ya, yb)

        # Normalize the joint density
        normalized_jpdf = (joint_density - self.min_jpdf) / (self.max_jpdf - self.min_jpdf)

        # Compute the reweighting factor
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
        # Assume self.kde is already a gaussian_kde object fitted with y_train data
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

    def __init__(self, X, y, alpha: float = .9,
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

        # Create training data
        self.X_train = X
        self.y_train = y

        self.min_y = np.min(self.y_train)
        self.max_y = np.max(self.y_train)

        self.kde = gaussian_kde(self.y_train, bw_method='scott')
        self.reweights = self.preprocess_reweighting(self.y_train)  # for labels, order maintained

        if self.debug:
            print('X_train: ', self.X_train[:12])
            print('y_train: ', self.y_train[:12])
            print('reweights: ', self.reweights[:12])
            self.plot_density_kde_reweights()

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

    def plot_density_kde_reweights(self):
        """
        Plot the label density, KDE, and reweights for the y_train dataset.
        """
        # Create a range of y values for plotting
        y_values = np.linspace(self.min_y, self.max_y, 1000)

        # Compute density using the histogram method
        hist_density, bin_edges = np.histogram(self.y_train, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Compute KDE values
        kde_values = self.kde.evaluate(y_values)

        # Get reweights for the plotting range (you can substitute this with your reweights)
        reweights_plot = self.normalized_reweight(y_values, self.alpha)  # Assuming normalized_reweight method exists

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, hist_density, label='Label Density', color='blue')
        plt.plot(y_values, kde_values, label='KDE', color='green')
        plt.plot(y_values, reweights_plot, label='Reweights', color='red')
        plt.xlabel('y_values')
        plt.ylabel('Density / Reweights')
        plt.legend()
        plt.title('Label Density, KDE, and Reweights')
        plt.show()

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

        :param y: The y-value of the data points as a NumPy array.
        :param alpha: Parameter to adjust the reweighting.
        :param epsilon: A small constant to avoid zero reweighting.
        :return: The reweighting factor for the label as a NumPy array.
        """
        # Compute the density of y
        density = self.pdf(y)

        # Normalize the joint density
        normalized_pdf = (density - self.min_pdf) / (self.max_pdf - self.min_pdf)

        # Compute the reweighting factor
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
