##############################################################################################################
# Description: generating datasets for ML based on csv files you received
# (features might be added/removed over time, training/validation/test splits)
##############################################################################################################

# imports
import numpy as np
from scipy.stats import gaussian_kde
import tensorflow as tf
import matplotlib.pyplot as plt
# types for type hinting
from typing import Tuple


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


    def __init__(self, n_train: int = 1000, n_test: int = 1000, n_features: int = 4, debug: bool = True) -> None:
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

        self.debug = debug

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
        self.alpha = 1
        self.reweights = self.preprocess_reweighting(self.y_train)

        # print 4 first rows of X_train, y_train, X_test, y_test
        if self.debug:
            print('X_train: ', self.X_train[:4])
            print('y_train: ', self.y_train[:4])
            print('X_test: ', self.X_test[:4])
            print('y_test: ', self.y_test[:4])
            print('min_val y in after norm: ', self.min_y)
            print('max_val y in after norm: ', self.max_y)
            self.plot_density_kde_reweights()

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

    def jpdf(self, ya: tf.Tensor, yb: tf.Tensor) -> tf.Tensor:
        """
        Joint Probability Density Function for two independent variables ya and yb.

        :param kde: The kernel density estimation object.
        :param ya: The first y value as a TensorFlow tensor.
        :param yb: The second y value as a TensorFlow tensor.
        :return: The product of the probability densities at ya and yb as a TensorFlow tensor.
        """

        def py_jpdf(ya, yb):
            # Density of ya and yb
            density_ya = self.kde.evaluate(ya)
            density_yb = self.kde.evaluate(yb)

            # Joint density
            joint_density = density_ya * density_yb

            return joint_density[0]

        return tf.py_function(func=py_jpdf, inp=[ya, yb], Tout=tf.float32)

    def pdf(self, y: np.ndarray) -> np.ndarray:
        """
        Probability Density Function for label y.
        :param y: The y value as a NumPy array.
        :return: The probability density at y as a NumPy array.
        """
        return self.kde.evaluate(y)

    def find_min_max_pdf(self, y: np.ndarray) -> None:
        """
        Find the minimum and maximum PDF values for a given NumPy array of labels y.

        :param y: A NumPy array containing labels.
        :return: None. Updates self.min_pdf and self.max_pdf.
        """
        pdf_values = self.pdf(y)

        self.min_pdf = np.min(pdf_values)
        self.max_pdf = np.max(pdf_values)

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


    def jreweight(self, ya: float, yb: float, alpha: float, epsilon: float = 1e-7) -> float:
        """
        Calculate the reweighting factor for a pair of labels ya and yb.

        :param ya: The y-value of the first data point.
        :param yb: The y-value of the second data point.
        :param min_jpdf: The minimum jpdf value in the dataset.
        :param max_jpdf: The maximum jpdf value in the dataset.
        :param kde_model: Trained KDE model.
        :param alpha: Parameter to adjust the reweighting.
        :param epsilon: A small constant to avoid zero reweighting.
        :return: The reweighting factor for the pair ya, yb.
        """

        # Compute the joint density of ya and yb
        joint_density = self.jpdf(ya, yb)

        # Make sure joint_density is a tensor with proper shape and dtype
        joint_density = tf.ensure_shape(joint_density, ())

        # Normalize the joint density
        normalized_jpdf = (joint_density - self.min_jpdf) / (self.max_jpdf - self.min_jpdf)

        # Compute the reweighting factor
        reweighting_factor = tf.math.maximum(1 - alpha * normalized_jpdf, epsilon)

        return reweighting_factor
