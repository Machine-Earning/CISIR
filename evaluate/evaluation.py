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
from typing import Tuple, List, Optional, Any
from numpy import ndarray
from tensorflow.keras import Model
from sklearn.metrics import confusion_matrix, f1_score


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
                 threshold: float = 10, save_tag=None) -> float:
        """
        Evaluate the performance of the model on test data using TensorFlow's MSE and plot error per bin.

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
        lower_threshold = np.log(threshold_val / np.exp(2)) + 1e-6  # + 1e-9 to avoid backgrounds being considered

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

        return mae

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

    def check_norms(self, model: Model, X_test: np.ndarray) -> None:
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
