##############################################################################################################
# Description: training and testing (algos, nn structure, loss functions,
# using validation loss to determine epoch number for training).
# this module should be interchangeable with other modules (
##############################################################################################################

# imports
import tensorflow as tf
from tensorflow.keras import layers, callbacks, Model
import datetime
import numpy as np
import matplotlib.pyplot as plt

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
    lambda_coef = None

    def __init__(self, debug: bool = True) -> None:
        """
        Initialize the class variables.

        :param debug: Boolean to enable debug output.
        """
        self.debug = debug
        self.lambda_coef = 4  # coefficient for matching feature loss and regression loss

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
        repr_layer = NormalizeLayer(name='normalize_layer')(x)
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

    def add_regression_and_decoder_heads(self, base_model: Model) -> Model:
        """
        Add a regression head with one output unit and a decoder head that aims to reconstruct the input
        to an existing neural network model.

        :param base_model: The base neural network model.
        :return: The extended model with a regression head and a decoder head.
        """
        # Extract the output of the last layer of the base model
        last_layer_output = base_model.output

        # Add a Dense layer with one output unit for regression
        regression_head = layers.Dense(1, activation='linear', name="regression_head")(last_layer_output)

        # Add a decoder head for input reconstruction
        # Assuming the original input has shape (None, 4)
        decoder_head = layers.Dense(8, activation='linear', name="decoder_layer_1")(last_layer_output)
        decoder_head = layers.LeakyReLU()(decoder_head)
        decoder_head = layers.Dense(4, activation='linear', name="decoder_head")(decoder_head)

        # Create the new extended model
        extended_model = Model(inputs=base_model.input, outputs=[last_layer_output, regression_head, decoder_head])

        return extended_model

    def reset_regression_head(self, extended_model: Model) -> Model:
        """
        Reset the regression head of an extended neural network model.

        :param extended_model: The extended neural network model.
        :return: The base model without the regression head.
        """
        # Serialize the model to a config dictionary
        config = extended_model.get_config()

        # Remove the regression head layer from the config
        config['layers'].pop()

        # Remove the regression head from output_layers
        config['output_layers'] = [x for x in config['output_layers'] if x[0] != 'regression_head']

        # Provide the NormalizeLayer as custom object
        custom_objects = {'NormalizeLayer': NormalizeLayer}

        # Reconstruct the model from the config
        base_model = Model.from_config(config, custom_objects=custom_objects)

        # Copy weights for all layers except the last one (the regression head)
        for layer, old_layer in zip(base_model.layers, extended_model.layers[:-1]):
            layer.set_weights(old_layer.get_weights())

        return base_model

    def freeze_features(self, model: Model) -> None:
        """
        Freeze the weights of the base model so only the regression head is trainable.

        :param model: The neural network model with a regression head.
        """
        # Iterate through each layer and set it to be non-trainable
        for layer in model.layers:
            if layer.name != 'regression_head':  # Don't freeze the regression head
                layer.trainable = False

        # Explicitly set the regression head to be trainable
        regression_head = model.get_layer('regression_head')
        regression_head.trainable = True

    def train_jointly(self,
                      model: Model,
                      X_train: Tensor,
                      y_train: Tensor,
                      y_regression: Tensor,
                      sample_weights: np.ndarray = None,
                      sample_joint_weights: np.ndarray = None,
                      learning_rate: float = 1e-3,
                      epochs: int = 100,
                      batch_size: int = 32,
                      patience: int = 9,
                      lambda_coef: float = 4.0) -> callbacks.History:
        """
        Train a neural network model focusing on both the feature representation and regression output.

        :param model: The neural network model.
        :param X_train: Training features.
        :param y_train: Training labels .
        :param sample_weights: Sample weights for regression head.
        :param sample_joint_weights: Sample weights for feature representation.
        :param learning_rate: Learning rate for Adam optimizer.
        :param epochs: Number of epochs to train.
        :param batch_size: Size of batches during training.
        :param patience: Number of epochs to wait for early stopping.
        :param lambda_coef: Coefficient for balancing the two loss functions.
        :return: Training history.
        """

        class WeightedLossCallback(callbacks.Callback):
            def __init__(self, sample_joint_weights, y_shape):
                self.sample_joint_weights = sample_joint_weights
                self.y_shape = y_shape

            def on_train_batch_begin(self, batch, logs=None):
                if self.sample_joint_weights is not None:
                    idx1, idx2 = np.triu_indices(self.y_shape, k=1)
                    one_d_indices = np.ravel_multi_index((idx1, idx2), (self.y_shape, self.y_shape))
                    joint_weights_batch = self.sample_joint_weights[one_d_indices]
                    self.model.loss_weights = {'feature_head': joint_weights_batch, 'regression_head': lambda_coef}

        # Define the losses and their weights
        losses = {'feature_head': self.repr_loss, 'regression_head': 'mse'}
        loss_weights = {'feature_head': 1.0, 'regression_head': lambda_coef}

        # TensorBoard setup
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Early stopping setup
        early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Weighted loss callback setup
        weighted_loss_cb = WeightedLossCallback(sample_joint_weights, y_train.shape[0])

        # Callback list
        callback_list = [tensorboard_cb, early_stopping_cb]

        if sample_joint_weights is not None:
            callback_list.append(weighted_loss_cb)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=losses, loss_weights=loss_weights)

        # Fit the model
        history = model.fit(X_train, {'feature_head': y_train, 'regression_head': y_regression},
                            sample_weight={'feature_head': sample_joint_weights, 'regression_head': sample_weights},
                            epochs=epochs, batch_size=batch_size,
                            validation_split=0.3, callbacks=callback_list)

        # Get the best epoch
        best_epoch = np.argmin(history.history['val_loss']) + 1

        # Plotting the loss
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['feature_head_loss'], label='Feature Training Loss')
        plt.plot(history.history['val_feature_head_loss'], label='Feature Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Feature Head Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['regression_head_loss'], label='Regression Training Loss')
        plt.plot(history.history['val_regression_head_loss'], label='Regression Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Regression Head Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return history

    def train_features(self,
                       model: Model,
                       X_train: Tensor,
                       y_train: Tensor,
                       X_val: Tensor,
                       y_val: Tensor,
                       sample_joint_weights: np.ndarray = None,
                       learning_rate: float = 1e-3,
                       epochs: int = 100,
                       batch_size: int = 32,
                       patience: int = 9) -> callbacks.History:
        """
        Trains the model and returns the training history.

        :param model: The TensorFlow model to train.
        :param X_train: The training feature set.
        :param y_train: The training labels.
        :param X_val: Validation features.
        :param y_val: Validation labels.
        :param sample_joint_weights: The reweighting factors for pairs of labels.
        :param learning_rate: The learning rate for the Adam optimizer.
        :param epochs: The maximum number of epochs for training.
        :param batch_size: The batch size for training.
        :param patience: The number of epochs with no improvement to wait before early stopping.
        :return: The training history as a History object.
        """

        # Setup TensorBoard
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        print("Run the command line:\n tensorboard --logdir logs/fit")

        # Setup early stopping
        early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # In your Callback
        class WeightedLossCallback(callbacks.Callback):
            def on_train_batch_begin(self, batch, logs=None):
                idx1, idx2 = np.triu_indices(len(y_train), k=1)
                one_d_indices = [map_to_1D_idx(i, j, len(y_train)) for i, j in zip(idx1, idx2)]
                joint_weights_batch = sample_joint_weights[one_d_indices]  # Retrieve weights for this batch
                self.model.loss_weights = joint_weights_batch  # Set loss weights for this batch

        # Create an instance of the custom callback
        weighted_loss_cb = WeightedLossCallback()

        # Include weighted_loss_cb in callbacks only if sample_joint_weights is not None
        callback_list = [tensorboard_cb, early_stopping_cb]
        if sample_joint_weights is not None:
            callback_list.append(weighted_loss_cb)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=self.repr_loss)

        # First train the model with a validation set to determine the best epoch
        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=callback_list)

        # Get the best epoch from early stopping
        best_epoch = np.argmin(history.history['val_loss']) + 1

        # Plot training loss and validation loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.show()

        # Retrain the model on the combined dataset (training + validation) to the best epoch found
        X_combined = np.concatenate((X_train, X_val), axis=0)
        y_combined = np.concatenate((y_train, y_val), axis=0)

        if sample_joint_weights is not None:
            sample_joint_weights_combined = np.concatenate((sample_joint_weights, sample_joint_weights), axis=0)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=self.repr_loss)
        if sample_joint_weights is not None:
            model.fit(X_combined, y_combined, epochs=best_epoch, batch_size=batch_size, callbacks=[weighted_loss_cb])
        else:
            model.fit(X_combined, y_combined, epochs=best_epoch, batch_size=batch_size)


        return history

    def train_regression(self,
                         model: Model,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_val: np.ndarray,
                         y_val: np.ndarray,
                         sample_weights: Optional[np.ndarray] = None,
                         sample_val_weights: Optional[np.ndarray] = None,
                         learning_rate: float = 1e-3,
                         epochs: int = 100,
                         batch_size: int = 32,
                         patience: int = 9) -> callbacks.History:
        """
        Train a neural network model focusing only on the regression output.
        Include reweighting for balancing the loss.

        :param model: The neural network model.
        :param X_train: Training features.
        :param y_train: Training labels.
        :param X_val: Validation features.
        :param y_val: Validation labels.
        :param sample_weights: Sample weights for training set.
        :param sample_val_weights: Sample weights for validation set.
        :param learning_rate: Learning rate for Adam optimizer.
        :param epochs: Number of epochs.
        :param batch_size: Batch size.
        :param patience: Number of epochs for early stopping.
        :return: Training history.
        """

        # Setup TensorBoard
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        print("Run the command line:\n tensorboard --logdir logs/fit")

        # Early stopping callback
        early_stopping_cb = callbacks.EarlyStopping(monitor='val_regression_head_loss', patience=patience, restore_best_weights=True)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss={'regression_head': 'mse'})

        # Train the model with a validation set
        history = model.fit(X_train,  {'regression_head': y_train},
                            sample_weight=sample_weights,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, {'regression_head': y_val}, sample_val_weights),
                            callbacks=[tensorboard_cb, early_stopping_cb])

        # Find the best epoch from early stopping
        best_epoch = np.argmin(history.history['val_regression_head_loss']) + 1

        # Plot training and validation loss
        plt.plot(history.history['regression_head_loss'], label='Training Loss')
        plt.plot(history.history['val_regression_head_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.show()

        # Combine training and validation data for final training
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        # Combine sample weights if provided
        if sample_weights is not None and sample_val_weights is not None:
            full_sample_weights = np.concatenate([sample_weights, sample_val_weights], axis=0)
        else:
            full_sample_weights = None

        # Retrain the model to the best epoch using combined data
        model.fit(X_full, {'regression_head': y_full},
                  sample_weight=full_sample_weights,
                  epochs=best_epoch,
                  batch_size=batch_size,
                  callbacks=[tensorboard_cb])

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

    def repr_loss_fast(self, y_true, z_pred, reduction=tf.keras.losses.Reduction.NONE):
        """
        Computes the loss for a batch of predicted features and their labels.
        TODO: to fix, values are off

        :param y_true: A batch of true label values, shape of [batch_size, 1].
        :param z_pred: A batch of predicted Z values, shape of [batch_size, 2].
        :param reduction: The type of reduction to apply to the loss.
        :return: The average error for all unique combinations of the samples in the batch.
        """
        batch_size = tf.shape(z_pred)[0]
        denom = tf.cast(batch_size * (batch_size - 1) / 2, dtype=tf.float32)

        # Compute all pairs of errors at once
        # We expand dimensions to prepare for broadcasting
        z1 = tf.expand_dims(z_pred, 1)
        z2 = tf.expand_dims(z_pred, 0)
        label1 = tf.expand_dims(y_true, 1)
        label2 = tf.expand_dims(y_true, 0)

        # Compute the pairwise errors using the 'self.error' function
        err_matrix = self.error(z1, z2, label1, label2)

        # Exclude the diagonal (self-comparisons) and take upper triangle
        mask = 1 - tf.eye(batch_size, dtype=tf.float32)
        err_matrix *= mask  # Mask out diagonal
        total_error = tf.reduce_sum(err_matrix) / 2  # Sum over upper triangle elements

        if reduction == tf.keras.losses.Reduction.SUM:
            return total_error  # total loss
        elif reduction == tf.keras.losses.Reduction.NONE:
            return total_error / (denom + 1e-9)  # average loss
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}.")

    def repr_loss(self, y_true, z_pred, reduction=tf.keras.losses.Reduction.NONE):
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


def map_to_1D_idx(i, j, n):
    """Map the 2D index (i, j) of an n x n upper triangular matrix to the
    corresponding 1D index of its flattened form.
    :param i: The row index.
    :param j: The column index.
    :param n: The number of rows in the upper triangular matrix.
    :return: The 1D index of the flattened form.
    """
    return n * i + j - ((i + 1) * (i + 2)) // 2
