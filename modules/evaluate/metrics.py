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


class PCCPlusMetric(tf.keras.metrics.Metric):
    """
    Custom Pearson Correlation Coefficient (PCC+) metric that computes PCC for target values above a given threshold.

    Attributes:
    - threshold (float, optional): The threshold value. If None, the metric behaves like standard PCC.
    """
    def __init__(self, threshold: Optional[float] = None, name: str = "pcc_plus", **kwargs):
        super(PCCPlusMetric, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.y_true_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, infer_shape=False)
        self.y_pred_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, infer_shape=False)
        self.array_size = self.add_weight(name="array_size", initializer="zeros", dtype=tf.int32)

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

        # Write the filtered values to the TensorArray using the current size index
        self.y_true_array = self.y_true_array.write(self.array_size, y_true_filtered)
        self.y_pred_array = self.y_pred_array.write(self.array_size, y_pred_filtered)

        # Increment the size counter
        self.array_size.assign_add(1)

    def result(self) -> tf.Tensor:
        """
        Computes and returns the final PCC+ value.

        Returns:
        - tf.Tensor: The Pearson Correlation Coefficient across all batches.
        """
        # Concatenate the accumulated true and predicted values from the TensorArrays
        y_true = self.y_true_array.concat()
        y_pred = self.y_pred_array.concat()

        # Compute PCC using TensorFlow operations
        y_true_mean = tf.reduce_mean(y_true)
        y_pred_mean = tf.reduce_mean(y_pred)

        y_true_centered = y_true - y_true_mean
        y_pred_centered = y_pred - y_pred_mean

        covariance = tf.reduce_sum(y_true_centered * y_pred_centered)
        true_var = tf.reduce_sum(tf.square(y_true_centered))
        pred_var = tf.reduce_sum(tf.square(y_pred_centered))

        denominator = tf.sqrt(true_var) * tf.sqrt(pred_var)

        pcc_value = covariance / denominator
        return tf.where(tf.math.is_nan(pcc_value), 0.0, pcc_value)

    def reset_states(self):
        """
        Resets the metric state variables (TensorArrays) at the start of a new epoch.
        """
        self.y_true_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, infer_shape=False)
        self.y_pred_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, infer_shape=False)
        self.array_size.assign(0)  # Reset the size counter to 0
