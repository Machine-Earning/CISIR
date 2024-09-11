from typing import Optional

import tensorflow as tf


class MAEPlusMetric(tf.keras.metrics.Metric):
    """
    Custom Mean Absolute Error (MAE+) metric that computes MAE for target values above a given threshold.

    Attributes:
    - threshold (float, optional): The threshold value. If None, the metric behaves like standard MAE.
    - mae (tf.Variable): Accumulated MAE value.
    - count (tf.Variable): Count of batches contributing to the MAE calculation.
    """

    def __init__(self, threshold: Optional[float] = None, name: str = "mae_plus", **kwargs):
        super(MAEPlusMetric, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.mae = self.add_weight(name="mae", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor] = None):
        """
        Updates the state of the metric with new observations.

        Parameters:
        - y_true (tf.Tensor): True target values.
        - y_pred (tf.Tensor): Predicted values.
        - sample_weight (tf.Tensor, optional): Weights for the samples, if any.
        """
        # Filter based on threshold if provided, otherwise use all values
        if self.threshold is not None:
            mask = tf.greater_equal(y_true, self.threshold)
            y_true_filtered = tf.boolean_mask(y_true, mask)
            y_pred_filtered = tf.boolean_mask(y_pred, mask)
        else:
            y_true_filtered = y_true
            y_pred_filtered = y_pred

        # Compute the MAE for the filtered values
        mae_value = tf.reduce_mean(tf.abs(y_true_filtered - y_pred_filtered))

        # Accumulate MAE values and increment the count
        self.mae.assign_add(mae_value)
        self.count.assign_add(1)

    def result(self) -> tf.Tensor:
        """
        Computes and returns the final MAE+ value.

        Returns:
        - tf.Tensor: The average MAE+ across all batches.
        """
        return self.mae / self.count

    def reset_states(self):
        """
        Resets the metric state variables at the start of a new epoch.
        """
        self.mae.assign(0.0)
        self.count.assign(0.0)


from scipy.stats import pearsonr
from typing import Optional


class PCCPlusMetric(tf.keras.metrics.Metric):
    """
    Custom Pearson Correlation Coefficient (PCC+) metric that computes PCC for target values above a given threshold.

    Attributes:
    - threshold (float, optional): The threshold value. If None, the metric behaves like standard PCC.
    - y_true_list (List[tf.Tensor]): List accumulating true values across batches.
    - y_pred_list (List[tf.Tensor]): List accumulating predicted values across batches.
    """

    def __init__(self, threshold: Optional[float] = None, name: str = "pcc_plus", **kwargs):
        super(PCCPlusMetric, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.y_true_list = []
        self.y_pred_list = []

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor] = None):
        """
        Updates the state of the metric with new observations.

        Parameters:
        - y_true (tf.Tensor): True target values.
        - y_pred (tf.Tensor): Predicted values.
        - sample_weight (tf.Tensor, optional): Weights for the samples, if any.
        """
        # Filter based on threshold if provided, otherwise use all values
        if self.threshold is not None:
            mask = tf.greater_equal(y_true, self.threshold)
            y_true_filtered = tf.boolean_mask(y_true, mask)
            y_pred_filtered = tf.boolean_mask(y_pred, mask)
        else:
            y_true_filtered = y_true
            y_pred_filtered = y_pred

        # Append the filtered true and predicted values for later PCC calculation
        self.y_true_list.append(y_true_filtered)
        self.y_pred_list.append(y_pred_filtered)

    def result(self) -> tf.Tensor:
        """
        Computes and returns the final PCC+ value.

        Returns:
        - tf.Tensor: The Pearson Correlation Coefficient across all batches.
        """
        # Concatenate the accumulated true and predicted values
        y_true = tf.concat(self.y_true_list, axis=0).numpy()
        y_pred = tf.concat(self.y_pred_list, axis=0).numpy()

        # Compute PCC using scipy's pearsonr
        if len(y_true) > 0 and len(y_pred) > 0:
            pcc_value, _ = pearsonr(y_true.flatten(), y_pred.flatten())
            return tf.convert_to_tensor(pcc_value, dtype=tf.float32)
        else:
            return tf.convert_to_tensor(0.0, dtype=tf.float32)  # Return 0 if no values meet the threshold

    def reset_states(self):
        """
        Resets the metric state variables (lists) at the start of a new epoch.
        """
        self.y_true_list = []
        self.y_pred_list = []
