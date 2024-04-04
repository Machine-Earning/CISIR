##############################################################################################################
# Description: generating datasets for ML based on csv files you received
# (features might be added/removed over time, training/validation/test splits)
##############################################################################################################

# types for type hinting
from typing import Tuple, List, Union, Optional
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.stats import gaussian_kde
import tensorflow as tf


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


def compute_joint_hist_density(ya_values, yb_values, bins=30):
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


class exDenseJointReweightsGPU:
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
        print('starting joint reweighting on GPU')
        y_tensor = tf.constant(y)
        n = len(y)
        i, j = tf.meshgrid(tf.range(n), tf.range(n), indexing='ij')
        mask = tf.greater(i, j)  # Upper triangle mask
        i = tf.boolean_mask(i, mask)
        j = tf.boolean_mask(j, mask)
        self.ya, self.yb = tf.gather(y_tensor, i), tf.gather(y_tensor, j)  # Get the unique pairs of ya and yb

        # Step 2: Find min and max joint PDF values and store them
        print('finding min and max joint pdf')
        self.find_min_max_jpdf(self.ya, self.yb)

        # Step 3: Find average joint reweighting factor
        print('finding average joint reweight')
        self.find_avg_jreweight(self.ya, self.yb, self.alpha)

        # Step 4: Calculate normalized joint reweighting factors
        print('calculating normalized joint reweight')
        normalized_joint_factors = self.normalized_jreweight(self.ya, self.yb, self.alpha)
        # Create a list of index pairs corresponding to ya and yb
        index_pairs = list(zip(i.numpy(), j.numpy()))

        print('joint reweighting done')
        normalized_joint_factors = normalized_joint_factors.numpy()
        self.ya = self.ya.numpy()
        self.yb = self.yb.numpy()

        return normalized_joint_factors, index_pairs

    def normalized_jreweight(self, ya: tf.Tensor, yb: tf.Tensor, alpha: float) -> tf.Tensor:
        """
        Calculate the normalized reweighting factor for joint labels ya and yb.

        :param ya: The y-values as TensorFlow tensors.
        :param yb: The y-values as TensorFlow tensors.
        :param alpha: Parameter to adjust the reweighting.
        :return: The normalized reweighting factor for the labels as a TensorFlow tensor.
        """
        # Ensure average reweight is not zero to avoid division by zero
        if self.avg_jreweight == 0:
            raise ValueError("Average reweighting factor should not be zero.")

        jreweight_factor = self.jreweight(ya, yb, alpha)
        normalized_joint_factor = jreweight_factor / self.avg_jreweight

        return normalized_joint_factor

    def find_avg_jreweight(self, ya: tf.Tensor, yb: tf.Tensor, alpha: float):
        """
        Find the average reweighting factor for joint labels ya and yb.
        :param ya: labels.
        :param yb: labels.
        :param alpha: Parameter to adjust the reweighting.
        :return: The average reweighting factor.
        """

        total_jreweight = tf.reduce_sum(self.jreweight(ya, yb, alpha))
        count = tf.size(ya)

        self.avg_jreweight = total_jreweight / tf.cast(count, tf.float16) if count > 0 else 0

    def jreweight(self, ya: tf.Tensor, yb: tf.Tensor, alpha: float) -> tf.Tensor:
        """
        Calculate the reweighting factor for joint labels ya and yb.

        :param ya: The y-values of the cme_files points as TensorFlow tensors.
        :param yb: The y-values of the cme_files points as TensorFlow tensors.
        :param alpha: Parameter to adjust the reweighting.
        :return: The reweighting factor for the labels as a TensorFlow tensor.
        """
        # Compute the joint density
        joint_density = self.jpdf(ya, yb)

        # Normalize the joint density
        normalized_jpdf = (joint_density - self.min_jpdf) / (self.max_jpdf - self.min_jpdf)

        # Compute the reweighting factor
        jreweighting_factor = (1 / (normalized_jpdf + 1e-8)) ** alpha

        return jreweighting_factor

    def find_min_max_jpdf(self, ya: tf.Tensor, yb: tf.Tensor) -> None:
        """
        Find the minimum and maximum joint PDF values for a given TensorFlow tensor of labels ya and yb.

        :param ya: A TensorFlow tensor containing labels for the first variable.
        :param yb: A TensorFlow tensor containing labels for the second variable.
        :return: None. Updates self.min_jpdf and self.max_jpdf.
        """
        joint_pdf_values = self.jpdf(ya, yb)
        # NOTE: this assume ya and yb are augmented with
        # all the possible pairings of original ya and yb
        # and this ya and yb will admit duplicates

        self.min_jpdf = tf.reduce_min(joint_pdf_values)
        self.max_jpdf = tf.reduce_max(joint_pdf_values)

    def compute_joint_kde(self, ya_values, yb_values):
        """
        Computes the joint KDE for a set of ya and yb values.
        """
        # Assume self.kde is already a gaussian_kde object fitted with y_train cme_files
        kde_ya = self.kde.evaluate(ya_values)
        kde_yb = self.kde.evaluate(yb_values)
        return kde_ya * kde_yb

    def jpdf(self, ya: tf.Tensor, yb: tf.Tensor) -> tf.Tensor:
        """
        Joint Probability Density Function for labels ya and yb.

        :param ya: The y value for the first variable as a TensorFlow tensor.
        :param yb: The y value for the second variable as a TensorFlow tensor.
        :return: The joint probability density as a TensorFlow tensor.
        """
        kde_ya = tf.convert_to_tensor(self.kde.evaluate(ya.numpy()))
        kde_yb = tf.convert_to_tensor(self.kde.evaluate(yb.numpy()))
        return kde_ya * kde_yb

    def plot_density_kde_jreweights(self):
        """
        Plot the joint label density, joint KDE, and joint reweights as separate subplots.
        """
        y_values = np.linspace(self.min_y, self.max_y, 100)
        Y1, Y2 = np.meshgrid(y_values, y_values)

        ya_values = Y1.ravel()
        yb_values = Y2.ravel()

        joint_hist_density, X, Y = compute_joint_hist_density(ya_values, yb_values)
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
