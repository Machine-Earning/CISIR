# imports
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import layers


# TODO: double check if there is a bug in the normalization layer.
class NormalizeLayer(layers.Layer):
    def __init__(self, epsilon: float = 1e-9, **kwargs):
        """
        Initialization for the NormalizeLayer.

        :param epsilon: A small constant to prevent division by zero during normalization. Default is 1e-9.
        :param kwargs: Additional keyword arguments for the parent class.
        """
        self.epsilon = epsilon
        self.kernel = None  # Explicitly define kernel as None
        super(NormalizeLayer, self).__init__(**kwargs)


    def call(self, reprs: Tensor) -> Tensor:
        """
        Forward pass for the NormalizeLayer.

        :param reprs: Input tensor of shape [batch_size, ...].
        :return: Normalized input tensor of the same shape as inputs.
        """
        norm = tf.norm(reprs, axis=1, keepdims=True) + self.epsilon
        return reprs / norm

    def get_config(self) -> dict:
        """
        Returns the config of the layer. Contains the layer's configuration as a dict,
        including the `epsilon` parameter and the configurations of the parent class.

        :return: A dict containing the layer's configuration.
        """
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon,
        })
        return config
