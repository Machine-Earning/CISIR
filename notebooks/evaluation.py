##############################################################################################################
# Description: this file will be used for evaluation (metrics, plots, experiments,  ...)
# generally each configuration is run 5 times to smooth out different results from random initialization.
##############################################################################################################


# imports
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
# types for type hinting
from typing import Tuple, List, Optional
from numpy import ndarray
from tensorflow.keras import Model

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

    def evaluate(self, model: Model, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluate the performance of the model on test data using TensorFlow's MSE.

        :param model: model to test
        :param X_test: Test features as a NumPy array.
        :param y_test: Test labels for the regression output as a NumPy array.
        :return: Performance as a percentage based on MSE. Lower is better.
        """

        # Predict the y-values using the model
        y_pred = model.predict(X_test)

        # Assuming y_pred may have multiple outputs and you're interested in the regression head
        # If it's not the second output, adjust the index accordingly.
        if isinstance(y_pred, list) and len(y_pred) > 1:
            y_pred = y_pred[1]

        # Calculate the Mean Squared Error using TensorFlow
        mse = tf.keras.losses.MeanSquaredError()(y_test, y_pred).numpy()

        # Print the MSE
        print(f"Mean Squared Error: {mse}")

        # Convert MSE to a 'percentage' for easy interpretation
        # This is a custom representation, not a standard way to represent performance
        mse_percentage = (1 - mse) * 100

        return mse_percentage

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

    def plot_feature_space(self, model: Model, data: ndarray, labels: ndarray, withhead:bool = False, groups: Optional[List[int]] = None):
        """
        Plot the instances in the 2D feature space of the representation layer (Z).

        :param model: Trained neural network model.
        :param data: Input data, shape of [num_instances, num_features].
        :param labels: Corresponding labels for the data.
        :param groups: Optional grouping for labels to define color assignment.
        """

        if withhead:
            repr_layer_model = Model(inputs=model.input, outputs=model.output[0])
        else:
            # Extract the representation layer from the model
            repr_layer_model = Model(inputs=model.input, outputs=model.output)

        # Predict the representation for the input data
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

    def check_norms(self, model: Model, X_test:np.ndarray) -> None:
        """
        Plot the L2-norms of the representation layer (Z) for the test data.
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
