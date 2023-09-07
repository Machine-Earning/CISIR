##############################################################################################################
# Description: training and testing (algos, nn structure, loss functions,
# using validation loss to determine epoch number for training).
# this module should be interchangeable with other modules (
##############################################################################################################

# imports
import tensorflow as tf
from tensorflow.keras import layers, callbacks, Model
import datetime


# types for type hinting
from typing import Tuple, List, Optional, Callable
from tensorflow import Tensor


class ModelBuilder:
    """
    Class for building a neural network model.
    """

    # class variables
    debug = False
    model = None
    hidden_arch = None
    input_size = None
    output_size = None

    def create_model(self, inputs: int, outputs: int, hiddens: List[int]) -> Model:
        """
        Create a neural network model with two branches using the Keras functional API.
        One branch will be used for fitting the output to the label, and the other will use the Z features for the
        custom loss.
        The Z features are normalized to fit on the unit circle.

        :param hiddens: List of integers representing the number of nodes in each hidden layer.
        :return: The uncompiled model.
        """
        inputs = layers.Input(shape=(inputs,))
        x = inputs
        # Define hidden layers according to architecture
        for nodes in hiddens:
            x = layers.Dense(nodes)(x)
            x = layers.LeakyReLU()(x)  # Replacing 'relu' activation with LeakyReLU
        # Define the representation layer (Z features)
        repr_layer = layers.Dense(2)(x)
        repr_layer = layers.LeakyReLU(name='repr_layer')(repr_layer)  # Replacing 'relu' activation with LeakyReLU
        # Normalize the representation layer to fit on the unit circle
        # repr_layer = Lambda(lambda z: z / tf.norm(z, axis=1, keepdims=True))(x)
        # Define the output layer for fitting to the label
        pred_output = layers.Dense(outputs, name='pred_output')(repr_layer)
        # Bundle the Z features and fitting output into a model
        self.model = Model(inputs=inputs, outputs=[pred_output])  # , repr_layer])

        return self.model

    def create_model_feat(self, inputs: int, feat_dim: int, hiddens: List[int]) -> Model:
        """
        Create a neural network model with two branches using the Keras functional API.
        One branch will be used for fitting the output to the label, and the other will use the Z features for the custom loss.
        The Z features are normalized to fit on the unit circle.

        :param hiddens: List of integers representing the number of nodes in each hidden layer.
        :return: The uncompiled model.
        """
        inputs = layers.Input(shape=(inputs,))
        x = inputs
        # Define hidden layers according to architecture
        for nodes in hiddens:
            x = layers.Dense(nodes)(x)
            x = layers.LeakyReLU()(x)
        # Define the representation layer (Z features)
        x = layers.Dense(feat_dim)(x)
        x = layers.LeakyReLU()(x)
        # Normalize the representation layer to fit on the unit circle
        repr_layer = NormalizeLayer()(x)
        # Bundle the Z features and fitting output into a model
        self.model = Model(inputs=inputs, outputs=repr_layer)

        return self.model

    def add_regression_head(self, base_model: Model) -> Model:
        """
        Add a regression head with one output unit to an existing neural network model.

        :param base_model: The base neural network model.
        :return: The extended model with a regression head.
        """
        # Extract the output of the last layer of the base model
        last_layer_output = base_model.output

        # Add a Dense layer with one output unit for regression
        regression_head = layers.Dense(1, activation='linear', name="regression_head")(last_layer_output)

        # Create the new extended model
        extended_model = Model(inputs=base_model.input, outputs=[last_layer_output, regression_head])

        return extended_model

    def train_model(self,
                    model: Model,
                    X_train: Tensor,
                    y_train: Tensor,
                    learning_rate: float = 1e-3,
                    epochs: int = 100,
                    batch_size: int = 32,
                    patience: int = 16) -> callbacks.History:
        """

        :param learning_rate:
        :param model:
        :param X_train:
        :param y_train:
        :param epochs:
        :param batch_size:
        :return:
        """

        # setup tensorboard
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        print("Run the command line:\n tensorboard --logdir logs/fit")
        early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=self.custom_loss)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                            callbacks=[tensorboard_cb, early_stopping_cb])

        return history

    def train_regression(self,
                         model: Model,
                         X_train: Tensor,
                         y_train: Tensor,
                         sample_weights: Tensor = None,
                         learning_rate: float = 1e-3,
                         epochs: int = 100,
                         batch_size: int = 32,
                         patience: int = 16) -> callbacks.History:
        """
        Train a neural network model focusing only on the regression output.
        Include reweighting for balancing the loss.

        :param sample_weights: sample weights to tackle imbalance
        :param learning_rate: Learning rate for Adam optimizer.
        :param model: The neural network model.
        :param X_train: Training features.
        :param y_train: Training labels for the regression output.
        :param epochs: Number of epochs to train.
        :param batch_size: Size of batches during training.
        :param patience: Number of epochs to wait for early stopping.
        :return: Training history.
        """

        # Setup TensorBoard
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        print("Run the command line:\n tensorboard --logdir logs/fit")

        # Early stopping callback
        early_stopping_cb = callbacks.EarlyStopping(monitor='val_regression_head_loss', patience=patience,
                                                    restore_best_weights=True)

        # Compile the model, focusing only on the regression head output for training
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss={'regression_head': 'mse'},
                      metrics={'regression_head': 'mse'})

        if sample_weights is None:
            history = model.fit(X_train, {'regression_head': y_train},
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=0.2,
                                callbacks=[tensorboard_cb, early_stopping_cb])
        else:
            # Train the model with sample weights
            history = model.fit(X_train, {'regression_head': y_train},
                                sample_weight=sample_weights,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=0.2,
                                callbacks=[tensorboard_cb, early_stopping_cb])

        return history

    def plot_model(self, model: Model) -> None:
        """
        Plot the model architecture and save the figure.

        :param model: The model to plot.
        :return: None
        """
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    def zdist(self, vec1: Tensor, vec2: Tensor) -> float:
        """
        Computes the squared L2 norm distance between two input feature vectors.

        :param vec1: The first input feature vector.
        :param vec2: The second input feature vector.
        :return: The squared L2 norm distance.
        """
        return tf.reduce_sum(tf.square(vec1 - vec2))

    def ydist(self, val1: float, val2: float) -> float:
        """
        Computes the squared distance between two label values.

        :param val1: The first label value.
        :param val2: The second label value.
        :return: The squared distance.
        """
        return (val1 - val2) ** 2

    def error(self, z1: Tensor, z2: Tensor, label1: float, label2: float) -> float:
        """
        Computes the error between the zdist of two input predicted z values and their ydist.
        Range of the error is [0, 16].

        :param z1: The predicted z value for the first input sample.
        :param z2: The predicted z value for the second input sample.
        :param label1: The label of the first input sample.
        :param label2: The label of the second input sample.
        :return: The squared difference between the zdist and ydist.
        """
        squared_difference = (self.zdist(z1, z2) - self.ydist(label1, label2)) ** 2
        return tf.reduce_sum(squared_difference)

    def custom_loss(self, y_true, z_pred, reduction=tf.keras.losses.Reduction.NONE):
        """
        Computes the loss for a batch of predicted features and their labels.

        :param y_true: A batch of true label values, shape of [batch_size, 1].
        :param z_pred: A batch of predicted Z values, shape of [batch_size, 2].
        :param reduction: The type of reduction to apply to the loss.
        :return: The average error for all unique combinations of the samples in the batch.
        """
        int_batch_size = tf.shape(z_pred)[0]
        batch_size = tf.cast(int_batch_size, dtype=tf.float32)
        total_error = tf.constant(0.0, dtype=tf.float32)

        # Loop through all unique pairs of samples in the batch
        for i in tf.range(int_batch_size):
            for j in tf.range(i + 1, int_batch_size):
                z1, z2 = z_pred[i], z_pred[j]
                # tf.print(z1, z2, sep=', ', end='\n')
                label1, label2 = y_true[i], y_true[j]
                # tf.print(label1, label2, sep=', ', end='\n')
                err = self.error(z1, z2, label1, label2)
                # tf.print(err, end='\n\n')
                total_error += tf.cast(err, dtype=tf.float32)

        # tf.print(total_error)

        if reduction == tf.keras.losses.Reduction.SUM:
            return total_error  # total loss
        elif reduction == tf.keras.losses.Reduction.NONE:
            denom = tf.cast(batch_size * (batch_size - 1) / 2 + 1e-9, dtype=tf.float32)
            # tf.print(denom)
            return total_error / denom  # average loss
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}.")





class NormalizeLayer(layers.Layer):
    def __init__(self, epsilon: float = 1e-9, **kwargs):
        """
        Initialization for the NormalizeLayer.

        :param epsilon: A small constant to prevent division by zero during normalization. Default is 1e-9.
        :param kwargs: Additional keyword arguments for the parent class.
        """
        self.epsilon = epsilon
        super(NormalizeLayer, self).__init__(**kwargs)

    def call(self, inputs: Tensor) -> Tensor:
        """
        Forward pass for the NormalizeLayer.

        :param inputs: Input tensor of shape [batch_size, ...].
        :return: Normalized input tensor of the same shape as inputs.
        """
        norm = tf.norm(inputs, axis=1, keepdims=True) + self.epsilon
        return inputs / norm

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


