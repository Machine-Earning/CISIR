from typing import Optional, Dict

import tensorflow as tf


def create_weight_tensor_fast(y_true: tf.Tensor, weight_dict: Optional[Dict[float, float]]) -> tf.Tensor:
    """
    Creates a tensor of weights corresponding to the values in y_true based on the provided weight_dict.

    Args:
        y_true (tf.Tensor): The tensor containing the ground truth labels.
        weight_dict (Dict[float, float], optional): A dictionary mapping label values to their corresponding weights.

    Returns:
        tf.Tensor: A tensor of weights corresponding to y_true.
    """
    if weight_dict is None:
        return tf.ones_like(y_true, dtype=tf.float32)
    # Convert the weight dictionary to sorted tensors
    unique_labels = tf.constant(sorted(weight_dict.keys()), dtype=tf.float32)
    weight_values = tf.constant([weight_dict[label] for label in sorted(weight_dict.keys())], dtype=tf.float32)
    # Flatten y_true if it has more than one dimension
    y_true_flat = tf.reshape(y_true, [-1])
    # Use tf.searchsorted to find the indices of y_true in unique_labels
    indices = tf.searchsorted(unique_labels, y_true_flat, side='left')
    # Handle the case where the search goes out of bounds
    num_unique_labels = tf.shape(unique_labels)[0]
    num_unique_labels = tf.cast(num_unique_labels, indices.dtype)  # Ensure the type matches
    indices = tf.clip_by_value(indices, 0, num_unique_labels - 1)
    # Gather the weights using the found indices
    y_true_weights = tf.gather(weight_values, indices)
    # Reshape the weights to match the original y_true shape if necessary
    y_true_weights = tf.reshape(y_true_weights, tf.shape(y_true))

    return y_true_weights


class TrainingPhaseManager:
    """
    Manages the training phase flag to switch between training and validation modes.
    This class encapsulates the `is_training` state, making it easier to integrate
    with the custom loss function and callback.
    """

    def __init__(self):
        self.is_training = True

    def set_training(self, is_training: bool) -> None:
        """
        Sets the current phase to training or validation.

        Args:
            is_training (bool): True if training phase, False if validation/testing phase.
        """
        self.is_training = is_training

    def is_training_phase(self) -> bool:
        """
        Returns whether the current phase is training.

        Returns:
            bool: True if in training phase, False otherwise.
        """
        return self.is_training


class IsTraining(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to update the training phase flag in the TrainingPhaseManager object.
    """

    def __init__(self, phase_manager: TrainingPhaseManager):
        """
        Initializes the callback with a reference to the TrainingPhaseManager.

        Args:
            phase_manager (TrainingPhaseManager): The manager that tracks the training phase.
        """
        super().__init__()
        self.phase_manager = phase_manager

    def on_train_batch_begin(self, batch, logs=None) -> None:
        """
        Called at the beginning of each training batch.
        """
        self.phase_manager.set_training(True)

    def on_test_batch_begin(self, batch, logs=None) -> None:
        """
        Called at the beginning of each validation batch.
        """
        self.phase_manager.set_training(False)
