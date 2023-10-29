##############################################################################################################
# Description: generating datasets for ML based on csv files you received
# (features might be added/removed over time, training/validation/test splits)
##############################################################################################################

# types for type hinting
from typing import Tuple

import matplotlib.pyplot as plt
# imports
import numpy as np
from numpy import ndarray
from scipy.stats import gaussian_kde


class DatasetGenerator:
    """
    Class for generating synthetic regression datasets.
    """

    # class variables
    debug = False
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    min_old_y = None
    max_old_y = None
    min_y = None
    max_y = None
    kde = None
    min_pdf = None
    max_pdf = None
    avg_reweight = None
    reweights = None
    alpha = None
    min_jpdf = None
    max_jpdf = None
    avg_jreweight = None
    jreweights = None

    def __init__(self, n_train: int = 1000, n_test: int = 1000, n_features: int = 4, alpha: float = 1,
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
        :return: X_train, y_train, X_test, y_test
        """

        self.yb = None
        self.ya = None
        self.debug = debug
        self.alpha = alpha

        # Create training data
        self.X_train = np.random.normal(loc=0, scale=1, size=(n_train, n_features))
        self.y_train = np.linalg.norm(self.X_train, axis=1)  # Compute the L2-norm
        # Create testing data
        self.X_test = np.random.normal(loc=0, scale=1, size=(n_test, n_features))
        self.y_test = np.linalg.norm(self.X_test, axis=1)  # Compute the L2-norm
        # normalize labels
        self.y_train = self.normalize_labels(self.y_train)
        self.y_test = self.normalize_labels(self.y_test)

        self.min_y = np.min(self.y_train)
        self.max_y = np.max(self.y_train)

        self.kde = gaussian_kde(self.y_train, bw_method='scott')
        self.reweights = self.preprocess_reweighting(self.y_train)  # for labels, order maintained
        # self.jreweights = self.preprocess_jreweighting(self.y_train)  # for pairs of labels

        self.X_val = np.empty((0, self.X_train.shape[1]))
        self.y_val = np.empty((0,))
        self.val_reweights = np.empty((0,))

        # get the validation set
        self.X_val, self.y_val, self.val_reweights = self.get_val_data(prob=.25)

        # print 12 first rows of X_train, y_train, X_test, y_test,  X_val, y_val
        if self.debug:
            print('X_train: ', self.X_train[:12])
            print("Validation X:", self.X_val[:12])
            print('y_train: ', self.y_train[:12])
            print("Validation y:", self.y_val[:12])
            print('X_test: ', self.X_test[:4])
            print('y_test: ', self.y_test[:4])
            print('min_val y in after norm: ', self.min_y)
            print('max_val y in after norm: ', self.max_y)
            # self.plot_density_kde_reweights()
            # self.plot_density_kde_jreweights()

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

    def get_val_data(self, prob: float = .25):
        """
        Get validation data using stratified sampling.
        Sort the labels in descending order, and randomly pick one out of every 1/prob.
        Also sorts the sample weights and carries them to the validation data.

        :param prob: probability of picking out a label
        :return: X_val, y_val, val_reweights
        """
        # Sort labels and weights in descending order
        sorted_idx = np.argsort(self.y_train)[::-1]
        sorted_y_train = self.y_train[sorted_idx]
        sorted_X_train = self.X_train[sorted_idx]
        sorted_reweights = self.reweights[sorted_idx]

        # Calculate the number of samples and step size for picking validation samples
        n = len(self.y_train)
        stepsize = int(1 / prob)

        # Initialize validation arrays and weights
        self.X_val = np.empty((0, sorted_X_train.shape[1]))
        self.y_val = np.empty((0,))
        self.val_reweights = np.empty((0,))

        # To keep track of the indices we've picked for the validation set in original arrays
        original_val_idx = []

        # Iterate over sorted labels and pick one random sample every 1/prob samples
        for i in range(0, n, stepsize):
            upper_bound = min(i + stepsize, n)
            idx = np.random.randint(i, upper_bound)

            # Add picked sample and weight to the validation set
            self.X_val = np.concatenate((self.X_val, [sorted_X_train[idx]]), axis=0)
            self.y_val = np.concatenate((self.y_val, [sorted_y_train[idx]]), axis=0)
            self.val_reweights = np.concatenate((self.val_reweights, [sorted_reweights[idx]]), axis=0)

            # Store the original index for removing later
            original_val_idx.append(sorted_idx[idx])

        # Remove validation samples and their weights from the training set
        keep_idx = np.setdiff1d(np.arange(n), np.array(original_val_idx))
        self.X_train = self.X_train[keep_idx]
        self.y_train = self.y_train[keep_idx]
        self.reweights = self.reweights[keep_idx]

        # self.plot_distributions(self.y_train, self.y_val)

        return self.X_val, self.y_val, self.val_reweights

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

    def compute_joint_kde(self, ya_values, yb_values):
        """
        Computes the joint KDE for a set of ya and yb values.
        """
        # Assume self.kde is already a gaussian_kde object fitted with y_train data
        kde_ya = self.kde.evaluate(ya_values)
        kde_yb = self.kde.evaluate(yb_values)
        return kde_ya * kde_yb

    def create_synthetic_data(self, n_train: int = 1000, n_test: int = 1000, n_features: int = 4) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a synthetic regression dataset.
        The input features (X) are randomly generated using a normal distribution centered at 0 with a standard
        deviation of 1.
        The target values (y) are calculated as the L2 norm (Euclidean norm) of the input features.

        :param n_train: Number of training instances.
        :param n_test: Number of testing instances.
        :param n_features: Number of input features.
        :return: X_train, y_train, X_test, y_test
        """

        # Create training data
        self.X_train = np.random.normal(loc=0, scale=1, size=(n_train, n_features))
        self.y_train = np.linalg.norm(self.X_train, axis=1)  # Compute the L2-norm
        # Create testing data
        self.X_test = np.random.normal(loc=0, scale=1, size=(n_test, n_features))
        self.y_test = np.linalg.norm(self.X_test, axis=1)  # Compute the L2-norm
        # normalize labels
        self.y_train = self.normalize_labels(self.y_train)
        self.y_test = self.normalize_labels(self.y_test)

        return self.X_train, self.y_train, self.X_test, self.y_test

    def normalize_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Normalize the target values so they are between 0 and 2 using the equation:
        y' = 2 * (y - min) / (max - min)

        :param y: The original target values.
        :return: The normalized target values.
        """
        # print shape of y
        if self.debug: print(y.shape)
        self.min_old_y = np.min(y)
        if self.debug: print('min_val y in before norm: ', self.min_old_y)
        self.max_old_y = np.max(y)
        if self.debug: print('max_val y in before norm: ', self.max_old_y)

        # Normalize the labels
        normalized_y = 2 * (y - self.min_old_y) / (self.max_old_y - self.min_old_y)

        return normalized_y

    def augment_data_by_perturb(self, X: np.ndarray, y: np.ndarray, threshold: float = 1.5, n_augment: int = 500,
                                perturbation_scale: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment the dataset by generating synthetic points through random perturbations of existing points
        with L2 norm > threshold.

        :param X: Input features, shape [n_samples, n_features].
        :param y: Target values, shape [n_samples].
        :param threshold: L2 norm threshold for identifying rare samples to augment.
        :param n_augment: Number of synthetic points to generate.
        :param perturbation_scale: Scale factor for random perturbations.
        :return: Augmented dataset (X_augmented, y_augmented).
        """
        # Identify indices of points with L2 norm greater than the threshold
        high_norm_indices = np.where(y > threshold)[0]

        # Initialize lists to store augmented data
        X_augment = []
        y_augment = []

        # Generate synthetic points
        for _ in range(n_augment):
            # Randomly choose a data point from the high_norm_indices
            idx = np.random.choice(high_norm_indices, 1)[0]

            # Generate a random perturbation
            perturbation = np.random.normal(scale=perturbation_scale, size=X[idx].shape)

            # Create perturbed point
            x_new = X[idx] + perturbation

            # Calculate its corresponding y value (L2 norm)
            y_new = np.linalg.norm(x_new)

            X_augment.append(x_new)
            y_augment.append(y_new)

        # Convert lists to NumPy arrays
        X_augment = np.array(X_augment)
        y_augment = np.array(y_augment)

        # Normalize the augmented labels
        y_augment = self.normalize_labels(y_augment)

        # Combine original and augmented data
        X_augmented = np.vstack((X, X_augment))
        y_augmented = np.hstack((y, y_augment))

        return X_augmented, y_augmented

    def augment_data_by_interpol(self, X: np.ndarray, y: np.ndarray, threshold: float = 1.5, n_augment: int = 500) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Augment the dataset by generating synthetic points interpolated between existing points
        with L2 norm > threshold. More samples will be generated closer to 2 than to the threshold.

        :param X: Input features, shape [n_samples, n_features].
        :param y: Target values, shape [n_samples].
        :param threshold: L2 norm threshold for identifying rare samples to augment.
        :param n_augment: Number of synthetic points to generate.
        :return: Augmented dataset (X_augmented, y_augmented).
        """
        # Identify indices of points with L2 norm greater than the threshold
        high_norm_indices = np.where(y > threshold)[0]
        high_norm_values = y[high_norm_indices]

        # Calculate selection probabilities proportional to distance from the threshold
        selection_probabilities = high_norm_values - threshold
        selection_probabilities /= selection_probabilities.sum()

        # Initialize lists to store augmented data
        X_augment = []
        y_augment = []

        # Generate synthetic points
        for _ in range(n_augment):
            # Randomly choose two data points from the high_norm_indices, weighted by selection_probabilities
            idx1, idx2 = np.random.choice(high_norm_indices, 2, replace=False, p=selection_probabilities)

            # Generate a random interpolation factor
            alpha = np.random.rand()

            # Create synthetic point and corresponding label
            x_new = alpha * X[idx1] + (1 - alpha) * X[idx2]
            y_new = alpha * y[idx1] + (1 - alpha) * y[idx2]

            X_augment.append(x_new)
            y_augment.append(y_new)

        # Convert lists to NumPy arrays
        X_augment = np.array(X_augment)
        y_augment = np.array(y_augment)

        # Normalize the augmented labels
        y_augment = self.normalize_labels(y_augment)

        # Combine original and augmented data
        X_augmented = np.vstack((X, X_augment))
        y_augmented = np.hstack((y, y_augment))

        return X_augmented, y_augmented

    def jpdf(self, ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        """
        Joint Probability Density Function for labels ya and yb.

        :param ya: The y value for the first variable as a NumPy array.
        :param yb: The y value for the second variable as a NumPy array.
        :return: The joint probability density as a NumPy array.
        """
        return self.kde.evaluate(ya) * self.kde.evaluate(yb)

    def pdf(self, y: np.ndarray) -> np.ndarray:
        """
        Probability Density Function for label y.
        :param y: The y value as a NumPy array.
        :return: The probability density at y as a NumPy array.
        """
        return self.kde.evaluate(y)

    def find_min_max_jpdf(self, ya: np.ndarray, yb: np.ndarray) -> None:
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

    def find_min_max_pdf(self, y: np.ndarray) -> None:
        """
        Find the minimum and maximum PDF values for a given NumPy array of labels y.

        :param y: A NumPy array containing labels.
        :return: None. Updates self.min_pdf and self.max_pdf.
        """
        pdf_values = self.pdf(y)

        self.min_pdf = np.min(pdf_values)
        self.max_pdf = np.max(pdf_values)

    def jreweight(self, ya: np.ndarray, yb: np.ndarray, alpha: float, epsilon: float = 1e-7) -> np.ndarray:
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

    def reweight(self, y: np.ndarray, alpha: float, epsilon: float = 1e-7) -> np.ndarray:
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

    def find_avg_jreweight(self, ya: np.ndarray, yb: np.ndarray, alpha: float, epsilon: float = 1e-7) -> float:
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

    def find_avg_reweight(self, y: np.ndarray, alpha: float, epsilon: float = 1e-7) -> float:
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

    def normalized_jreweight(self, ya: np.ndarray, yb: np.ndarray, alpha: float, epsilon: float = 1e-7) -> np.ndarray:
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

    def normalized_reweight(self, y: np.ndarray, alpha: float, epsilon: float = 1e-7) -> np.ndarray:
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

    def preprocess_jreweighting(self, y: np.ndarray) -> ndarray:
        """
        Preprocess reweighting for joint PDF based on a single dataset y and
        stores the unique pairs of ya and yb labels in self.ya and self.yb.

        :param y: The target dataset as a NumPy array.
        :return: None. Populates self.ya and self.yb with unique pairs.
        """

        # Step 1: Find all unique pairs of ya and yb
        # indices = range(len(y))
        # unique_pairs = list(combinations(indices, 2)) # NOT efficient
        # self.ya, self.yb = zip(*[(y[i], y[j]) for i, j in unique_pairs])
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

        return normalized_joint_factors

    def preprocess_reweighting(self, y: np.ndarray) -> np.ndarray:
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
