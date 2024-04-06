##############################################################################################################
# Description: training and testing (algos, nn structure, loss functions,
# using validation loss to determine epoch number for training).
# this module should be interchangeable with other modules (
##############################################################################################################
import time
from itertools import cycle
# types for type hinting
from typing import Tuple, List, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
# imports
import tensorflow as tf
from numpy import ndarray
from tensorflow import Tensor
from tensorflow.keras import layers, callbacks, Model
from tensorflow.keras.layers import (
    Input,
    Flatten,
    Dense,
    Concatenate,
    Dropout,
    LeakyReLU,
    BatchNormalization,
    LayerNormalization,
)

def ydist(val1: float, val2: float) -> float:
    """
    Computes the squared distance between two label values.

    :param val1: The first label value.
    :param val2: The second label value.
    :return: The squared distance.
    """
    return (val1 - val2) ** 2


def zdist(vec1: Tensor, vec2: Tensor) -> float:
    """
    Computes the squared L2 norm distance between two input feature vectors.

    :param vec1: The first input feature vector.
    :param vec2: The second input feature vector.
    :return: The squared L2 norm distance.
    """
    return tf.reduce_sum(tf.square(vec1 - vec2))


def error(z1: Tensor, z2: Tensor, label1: float, label2: float) -> Tensor:
    """
    Computes the error between the zdist of two input predicted z values and their ydist.
    Range of the error is [0, 8].

    :param z1: The predicted z value for the first input sample.
    :param z2: The predicted z value for the second input sample.
    :param label1: The label of the first input sample.
    :param label2: The label of the second input sample.
    :return: The squared difference between the zdist and ydist.
    """
    squared_difference = .5 * (zdist(z1, z2) - ydist(label1, label2)) ** 2
    # added multiplication by .5 to reduce the error range to 0-8
    return tf.reduce_sum(squared_difference)


class ModelBuilder:
    """
    Class for building a neural network model.
    """

    # class variables
    debug = False

    def __init__(self, debug: bool = True) -> None:
        """
        Initialize the class variables.

        :param debug: Boolean to enable debug output.
        """
        self.debug = debug
        self.sep_sep_count = tf.Variable(0, dtype=tf.int32)
        self.sep_elevated_count = tf.Variable(0, dtype=tf.int32)
        self.sep_background_count = tf.Variable(0, dtype=tf.int32)
        self.elevated_elevated_count = tf.Variable(0, dtype=tf.int32)
        self.elevated_background_count = tf.Variable(0, dtype=tf.int32)
        self.background_background_count = tf.Variable(0, dtype=tf.int32)
        self.number_of_batches = 0

    def create_model(self,
                     input_dim: int,
                     repr_dim: int,
                     output_dim: int,
                     hiddens: List[int],
                     with_ae: bool = False) -> Model:
        """
        Create a neural network model with options for multiple heads using the Keras functional API.

        :param input_dim: Integer representing the number of input features.
        :param repr_dim: Integer representing the dimensionality of the feature (representation layer).
        :param output_dim: Integer representing the dimensionality of the output.
        :param hiddens: List of integers representing the number of nodes in each hidden layer.
        :param with_ae: Boolean flag to indicate whether to include an AutoEncoder (AE) head for input reconstruction.

        :return: The uncompiled model.
        """
        # Input layer
        input_layer = layers.Input(shape=(input_dim,))
        x = input_layer

        # Define hidden layers according to architecture
        for nodes in hiddens:
            x = layers.Dense(nodes)(x)
            x = layers.LeakyReLU()(x)

        # Define the representation layer (Z features)
        repr_layer = layers.Dense(repr_dim)(x)
        repr_layer = layers.LeakyReLU(name='repr_layer')(repr_layer)

        # Add a regression head
        regression_head = layers.Dense(output_dim, activation='linear', name='regression_head')(repr_layer)

        # Create output_dim list
        outputs_list = [repr_layer, regression_head]

        # Add a decoder (AE) head for input reconstruction if with_ae is True
        if with_ae:
            decoder_head = repr_layer
            for nodes in reversed(hiddens):
                decoder_head = layers.Dense(nodes)(decoder_head)
                decoder_head = layers.LeakyReLU()(decoder_head)
            decoder_head = layers.Dense(input_dim, activation='linear', name='decoder_head')(decoder_head)
            outputs_list.append(decoder_head)

        # Create the model, repr, reg, decoder
        model = Model(inputs=input_layer, outputs=outputs_list)

        return model

    def create_model_pds(self,
                         input_dim: int,
                         repr_dim: int,
                         hiddens: List[int],
                         output_dim: Optional[int] = 1,
                         with_reg: bool = False, with_ae: bool = False) -> Model:
        """
        Create a neural network model with optional autoencoder and regression heads.
        The base model is used for feature extraction.

        :param input_dim: Integer representing the number of input features.
        :param repr_dim: Integer representing the dimensionality of the feature (representation layer).
        :param hiddens: List of integers representing the number of nodes in each hidden layer of the encoder.
        :param output_dim: Integer representing the dimensionality of the regression output. Default is 1.
        :param with_reg: Boolean flag to add a regression head to the model. Default is False.
        :param with_ae: Boolean flag to add a decoder to the model (making it an autoencoder). Default is False.
        :return: The uncompiled model with optional heads based on flags.
        """
        # Encoder
        encoder_input = layers.Input(shape=(input_dim,))
        x = encoder_input
        for nodes in hiddens:
            x = layers.Dense(nodes)(x)
            x = layers.LeakyReLU()(x)

        x = layers.Dense(repr_dim)(x)
        x = layers.LeakyReLU()(x)
        repr_layer = NormalizeLayer(name='normalize_layer')(x)

        outputs = [repr_layer]

        # Optional Regression Head
        if with_reg:
            x_reg = repr_layer
            regression_output = layers.Dense(output_dim, activation='linear', name='regression_head')(x_reg)
            outputs.append(regression_output)

        # Optional Decoder
        if with_ae:
            x_dec = repr_layer
            for nodes in reversed(hiddens):
                x_dec = layers.Dense(nodes)(x_dec)
                x_dec = layers.LeakyReLU()(x_dec)
            decoder_output = layers.Dense(input_dim, activation='linear', name='decoder_head')(x_dec)
            outputs.append(decoder_output)

        # Complete model, repr, reg, decoder
        model = Model(inputs=encoder_input, outputs=outputs if len(outputs) > 1 else outputs[0])

        return model

    # def add_proj_head(self,
    #                   model: Model,
    #                   output_dim: int = 1,
    #                   hiddens: Optional[List[int]] = None,
    #                   freeze_features: bool = True,
    #                   pds: bool = False) -> Model:
    #     """
    #     Add a regression head with one output unit and a projection layer to an existing model,
    #     replacing the existing prediction layer and optionally the decoder layer.
    #
    #     :param model: The existing model
    #     :param output_dim: The dimensionality of the output of the regression head.
    #     :param freeze_features: Whether to freeze the layers of the base model or not.
    #     :param hiddens: List of integers representing the hidden layers for the projection.
    #     :param pds: Whether to adapt the model for PDS representations.
    #     :return: The modified model with a projection layer and a regression head.
    #     """
    #
    #     if hiddens is None:
    #         hiddens = [6]
    #
    #     print(f'Features are frozen: {freeze_features}')
    #
    #     # Determine the layer to be kept based on whether PDS representations are used
    #     layer_to_keep = 'normalize_layer' if pds else 'repr_layer'
    #
    #     # Remove the last layer(s) to keep only the representation layer
    #     new_base_model = Model(inputs=model.input, outputs=model.get_layer(layer_to_keep).output)
    #
    #     # If freeze_features is True, freeze the layers of the new base model
    #     if freeze_features:
    #         for layer in new_base_model.layers:
    #             layer.trainable = False
    #
    #     # Extract the output of the last layer of the new base model (representation layer)
    #     repr_output = new_base_model.output
    #
    #     # Projection Layer(s)
    #     x_proj = repr_output
    #     for i, nodes in enumerate(hiddens):
    #         x_proj = layers.Dense(nodes, name=f"projection_layer_{i + 1}")(x_proj)
    #         x_proj = layers.LeakyReLU(name=f"projection_activation_{i + 1}")(x_proj)
    #
    #     # Add a Dense layer with one output unit for regression
    #     output_layer = layers.Dense(output_dim, activation='linear', name="forecast_head")(x_proj)
    #
    #     # Create the new extended model
    #     extended_model = Model(inputs=new_base_model.input, outputs=[repr_output, output_layer])
    #
    #     # If freeze_features is False, make all layers trainable
    #     if not freeze_features:
    #         for layer in extended_model.layers:
    #             layer.trainable = True
    #
    #     return extended_model

    def add_proj_head(self,
                      model: Model,
                      output_dim: int = 1,
                      hiddens: Optional[List[int]] = None,
                      freeze_features: bool = True,
                      pds: bool = False,
                      l2_reg: float = None,
                      dropout_rate: float = 0.0,
                      activation=None,
                      norm: str = None,
                      name: str = 'mlp') -> Model:
        """
        Add a regression head with one output unit and a projection layer to an existing model,
        replacing the existing prediction layer and optionally the decoder layer.

        :param model: The existing model
        :param output_dim: The dimensionality of the output of the regression head.
        :param freeze_features: Whether to freeze the layers of the base model or not.
        :param hiddens: List of integers representing the hidden layers for the projection.
        :param pds: Whether to adapt the model for PDS representations.
        :param l2_reg: L2 regularization factor.
        :param dropout_rate: Dropout rate for adding dropout layers.
        :param activation: Activation function to use. If None, defaults to LeakyReLU.
        :param norm: Type of normalization ('batch_norm' or 'layer_norm').
        :param name: Name of the model.
        :return: The modified model with a projection layer and a regression head.
        """

        if hiddens is None:
            hiddens = [6]

        if activation is None:
            activation = LeakyReLU()

        print(f'Features are frozen: {freeze_features}')

        # Determine the layer to be kept based on whether PDS representations are used
        layer_to_keep = 'normalize_layer' if pds else 'repr_layer'

        # Remove the last layer(s) to keep only the representation layer
        new_base_model = Model(inputs=model.input, outputs=model.get_layer(layer_to_keep).output)

        # If freeze_features is True, freeze the layers of the new base model
        if freeze_features:
            for layer in new_base_model.layers:
                layer.trainable = False

        # Count existing dropout layers to avoid naming conflicts
        dropout_count = sum(1 for layer in model.layers if isinstance(layer, Dropout))

        # Extract the output of the last layer of the new base model (representation layer)
        repr_output = new_base_model.output

        # Projection Layer(s)
        x_proj = repr_output
        for i, nodes in enumerate(hiddens):
            x_proj = Dense(
                nodes, kernel_regularizer=l2(l2_reg) if l2_reg else None,
                name=f"projection_layer_{i + 1}")(x_proj)

            if norm == 'batch_norm':
                x_proj = BatchNormalization(name=f"batch_norm_{i + 1}")(x_proj)
            elif norm == 'layer_norm':
                x_proj = LayerNormalization(name=f"layer_norm_{i + 1}")(x_proj)

            if callable(activation):
                x_proj = activation(x_proj)
            else:
                x_proj = LeakyReLU(name=f"activation_{i + 1}")(x_proj)

            if dropout_rate > 0.0:
                x_proj = Dropout(dropout_rate, name=f"proj_dropout_{dropout_count + i + 1}")(x_proj)

        # Add a Dense layer with one output unit for regression
        output_layer = Dense(output_dim, activation='linear', name=f"forecast_head")(x_proj)

        # Create the new extended model
        extended_model = Model(inputs=new_base_model.input, outputs=[repr_output, output_layer], name=name)

        # If freeze_features is False, make all layers trainable
        if not freeze_features:
            for layer in extended_model.layers:
                layer.trainable = True

        return extended_model

    def train_pds(self,
                  model: Model,
                  X_subtrain: ndarray,
                  y_subtrain: ndarray,
                  X_val: ndarray,
                  y_val: ndarray,
                  X_train: ndarray,
                  y_train: ndarray,
                  learning_rate: float = 1e-3,
                  epochs: int = 100,
                  batch_size: int = 32,
                  patience: int = 9,
                  save_tag=None,
                  callbacks_list=None,
                  verbose: int = 1) -> callbacks.History:
        """
        Trains the model and returns the training history.

        :param X_train: training and validation sets together
        :param y_train: labels of training and validation sets together
        :param save_tag: tag to use for saving experiments
        :param model: The TensorFlow model to train.
        :param X_subtrain: The training feature set.
        :param y_subtrain: The training labels.
        :param X_val: Validation features.
        :param y_val: Validation labels.
        :param learning_rate: The learning rate for the Adam optimizer.
        :param epochs: The maximum number of epochs for training.
        :param batch_size: The batch size for training.
        :param patience: The number of epochs with no improvement to wait before early stopping.
        :param callbacks_list: List of callback instances to apply during training.
        :param verbose: Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.


        :return: The training history as a History object.
        """

        if callbacks_list is None:
            callbacks_list = []

        # Initialize early stopping and model checkpointing
        early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                                    restore_best_weights=True)
        checkpoint_cb = callbacks.ModelCheckpoint(f"model_weights_{str(save_tag)}.h5", save_weights_only=True)

        # Append the early stopping and checkpoint callbacks to the custom callbacks list
        callbacks_list.extend([early_stopping_cb, checkpoint_cb])

        # Save initial weights for retraining on full training set after best epoch found
        initial_weights = model.get_weights()

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=self.pds_loss_vec)

        # First train the model with a validation set to determine the best epoch
        history = model.fit(X_subtrain, y_subtrain,
                            epochs=epochs,
                            batch_size=batch_size if batch_size > 0 else len(y_subtrain),
                            validation_data=(X_val, y_val),
                            validation_batch_size=batch_size if batch_size > 0 else len(y_val),
                            callbacks=callbacks_list,
                            verbose=verbose)

        # Get the best epoch from early stopping
        best_epoch = np.argmin(history.history['val_loss']) + 1

        # Plot training loss and validation loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        file_path = f"training_plot_{str(save_tag)}.png"
        plt.savefig(file_path)
        plt.close()

        # Reset model weights to initial state before retraining
        model.set_weights(initial_weights)

        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=self.pds_loss_vec)
        model.fit(X_train, y_train,
                  epochs=best_epoch,
                  batch_size=batch_size if batch_size > 0 else len(y_train),
                  callbacks=[checkpoint_cb],
                  verbose=verbose)

        # Evaluate the model on the entire training set
        # entire_training_loss = model.evaluate(X_train, y_train)

        # save the model weights
        model.save_weights(f"final_model_weights_{str(save_tag)}.h5")
        # print where the model weights are saved
        print(f"Model weights are saved in final_model_weights_{str(save_tag)}.h5")

        return history  # , entire_training_loss

    def train_pds_distributed(self,
                              model: Model,
                              final_model: Model,
                              subtrain_ds: tf.data.Dataset,
                              val_ds: tf.data.Dataset,
                              train_ds: tf.data.Dataset,
                              learning_rate: float = 1e-3,
                              epochs: int = 100,
                              patience: int = 9,
                              save_tag=None,
                              callbacks_list=None,
                              verbose: int = 1) -> callbacks.History:
        """
        Trains the model and returns the training history.
        TODO: needs to fix ASAP when worker > 1 and NAN happens

        :param subtrain_ds: The training feature set.
        :param val_ds: Validation features.
        :param train_ds: training and validation sets together
        :param save_tag: tag to use for saving experiments
        :param model: The TensorFlow model to subtrain for best epoch
        :param final_model: The TensorFlow model to train on the entire training set
        :param learning_rate: The learning rate for the Adam optimizer.
        :param epochs: The maximum number of epochs for training.
        :param patience: The number of epochs with no improvement to wait before early stopping.
        :param callbacks_list: List of callback instances to apply during training.
        :param verbose: Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.


        :return: The training history as a History object.
        """

        if callbacks_list is None:
            callbacks_list = []

        # Initialize early stopping and model checkpointing
        early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                                    restore_best_weights=True)
        checkpoint_cb = callbacks.ModelCheckpoint(f"model_weights_{str(save_tag)}.h5", save_weights_only=True)

        # Append the early stopping and checkpoint callbacks to the custom callbacks list
        callbacks_list.extend([early_stopping_cb, checkpoint_cb])

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=self.pds_loss_vec)

        # First train the model with a validation set to determine the best epoch
        history = model.fit(subtrain_ds,
                            epochs=epochs,
                            # batch_size=batch_size if batch_size > 0 else len(y_subtrain),
                            validation_data=val_ds,
                            # validation_batch_size=batch_size if batch_size > 0 else len(y_val),
                            callbacks=callbacks_list,
                            verbose=verbose)

        # Get the best epoch from early stopping
        best_epoch = np.argmin(history.history['val_loss']) + 1

        # Plot training loss and validation loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        file_path = f"training_plot_{str(save_tag)}.png"
        plt.savefig(file_path)
        plt.close()

        # Compile the final model
        final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=self.pds_loss_vec)

        final_model.fit(train_ds,
                        epochs=best_epoch,
                        # batch_size=batch_size if batch_size > 0 else len(y_train),
                        callbacks=[checkpoint_cb],
                        verbose=verbose)

        # Evaluate the model on the entire training set
        # entire_training_loss = model.evaluate(X_train, y_train)

        # save the model weights
        final_model.save_weights(f"model_weights_{str(save_tag)}.h5")
        # print where the model weights are saved
        print(f"Model weights are saved in model_weights_{str(save_tag)}.h5")

        return history  # , entire_training_loss

    def investigate_pds(self,
                        model: Model,
                        X_subtrain: ndarray,
                        y_subtrain: ndarray,
                        X_val: ndarray,
                        y_val: ndarray,
                        X_train: ndarray,
                        y_train: ndarray,
                        learning_rate: float = 1e-3,
                        epochs: int = 100,
                        batch_size: int = 32,
                        patience: int = 9,
                        save_tag=None) -> callbacks.History:
        """
        Trains the model and returns the training history.

        :param X_train:
        :param y_train:
        :param save_tag: tag to use for saving experiments
        :param model: The TensorFlow model to train.
        :param X_subtrain: The training feature set.
        :param y_subtrain: The training labels.
        :param X_val: Validation features.
        :param y_val: Validation labels.
        :param learning_rate: The learning rate for the Adam optimizer.
        :param epochs: The maximum number of epochs for training.
        :param batch_size: The batch size for training.
        :param patience: The number of epochs with no improvement to wait before early stopping.
        :return: The training history as a History object.
        """

        # Setup TensorBoard
        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        #
        # print("Run the command line:\n tensorboard --logdir logs/fit")

        # Initialize the custom callback
        investigate_cb = InvestigateCallback(model, X_train, y_train, batch_size, self, save_tag)

        # Setup early stopping
        early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # reduce learning rate on plateau
        # Initialize the ReduceLROnPlateau callback
        # reduce_lr_cb = callbacks.ReduceLROnPlateau(monitor='val_loss',
        #                                            factor=0.1,
        #                                            patience=5,
        #                                            min_lr=1e-6)
        # Setup model checkpointing
        checkpoint_cb = callbacks.ModelCheckpoint(f"model_weights_{str(save_tag)}.h5", save_weights_only=True)

        # Include weighted_loss_cb in callbacks only if sample_joint_weights is not None
        callback_list = [early_stopping_cb, checkpoint_cb]

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=self.pds_loss_vec)

        # First train the model with a validation set to determine the best epoch
        history = model.fit(X_subtrain, y_subtrain,
                            epochs=epochs,
                            batch_size=batch_size if batch_size > 0 else len(y_subtrain),
                            validation_data=(X_val, y_val),
                            validation_batch_size=batch_size if batch_size > 0 else len(y_val),
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
        file_path = f"training_plot_{str(save_tag)}.png"
        plt.savefig(file_path)
        plt.close()

        # Retrain the model on the combined dataset (training + validation) to the best epoch found
        # X_combined = np.concatenate((X_subtrain, X_val), axis=0)
        # y_combined = np.concatenate((y_subtrain, y_val), axis=0)

        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=self.pds_loss_vec)
        model.fit(X_train, y_train,
                  epochs=best_epoch,
                  batch_size=batch_size if batch_size > 0 else len(y_train),
                  callbacks=[checkpoint_cb, investigate_cb])
        # only investigates on the main training, not subtraining

        # Evaluate the model on the entire training set
        entire_training_loss = model.evaluate(X_train, y_train, batch_size=len(y_train))

        # save the model weights
        model.save_weights(f"model_weights_{str(save_tag)}.h5")

        return history, entire_training_loss

    def process_batch_weights(self, batch_indices: np.ndarray, label_weights_dict: Dict[float, float]) -> np.ndarray:
        """
        Process a batch of indices to return the corresponding joint weights.

        :param batch_indices: A batch of sample indices.
        :param label_weights_dict: Dictionary containing label weights.
        :return: An array containing joint weights corresponding to the batch of indices.
        """
        # Convert list of tuples into a dictionary for O(1) lookup
        weight_dict = {pair: weight for pair, weight in zip(joint_weight_indices, joint_weights)}

        batch_weights = []
        for i in batch_indices:
            for j in batch_indices:
                if i < j:  # Only consider pairs (i, j) where i < j
                    weight = label_weights_dict
                    if weight is not None:
                        batch_weights.append(weight)

        return np.array(batch_weights)

    # def process_batch_weights_vec(self,
    #                               batch_indices: np.ndarray,
    #                               joint_weights: np.ndarray,
    #                               joint_weight_indices: List[Tuple[int, int]]) -> np.ndarray:
    #     """
    #     Vectorized approach to return corresponding joint weights for upper diagonal pairs in the batch.
    #      TODO: speed up more
    #     :param batch_indices: A batch of sample indices.
    #     :param joint_weights: An array containing all joint weights for the dataset.
    #     :param joint_weight_indices: A list of tuples, each containing a pair of indices for which a joint weight exists.
    #     :return: An array containing joint weights corresponding to the upper diagonal pairs in the batch of indices.
    #     """
    #     # Create an efficient mapping from pairs to weights
    #     max_index = batch_indices.max() + 1
    #     weight_matrix = np.full((max_index, max_index), -1, dtype=int)  # Use an invalid index initially
    #
    #     # Populate the matrix with valid indices from joint_weight_indices
    #     for idx, (i, j) in enumerate(joint_weight_indices):
    #         if i < max_index and j < max_index:
    #             weight_matrix[i, j] = idx
    #
    #     # Generate upper diagonal pairs using broadcasting
    #     i_indices, j_indices = np.triu_indices_from(weight_matrix, k=1)
    #
    #     # Filter pairs that are in batch_indices
    #     valid_mask = np.isin(i_indices, batch_indices) & np.isin(j_indices, batch_indices)
    #     i_indices, j_indices = i_indices[valid_mask], j_indices[valid_mask]
    #
    #     # Lookup indices in the weight matrix
    #     weight_indices = weight_matrix[i_indices, j_indices]
    #
    #     # Filter out invalid indices
    #     valid_weight_indices = weight_indices[weight_indices != -1]
    #
    #     # Retrieve corresponding weights using the valid weight indices
    #     batch_weights = joint_weights[valid_weight_indices]
    #
    #     return batch_weights

    def train_for_one_epoch(self,
                            model: tf.keras.Model,
                            optimizer: tf.keras.optimizers.Optimizer,
                            loss_fn,
                            X: np.ndarray,
                            y: np.ndarray,
                            batch_size: int,
                            label_weights_dict: Optional[Dict[float, float]] = None,
                            training: bool = True) -> float:
        """
        Train or evaluate the model for one epoch.
        processing the batches with indices is what making it slow
        :param model: The model to train or evaluate.
        :param optimizer: The optimizer to use.
        :param loss_fn: The loss function to use.
        :param X: The feature set.
        :param y: The labels.
        :param batch_size: The batch size for training or evaluation.
        :param label_weights_dict: Dictionary containing label weights.
        :param training: Whether to apply training (True) or run evaluation (False).
        :return: The average loss for the epoch.
        """
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx in range(0, len(X), batch_size):
            batch_X = X[batch_idx:batch_idx + batch_size]
            batch_y = y[batch_idx:batch_idx + batch_size]

            if len(batch_y) <= 1:
                # can't form a pair so skip
                continue

            # print(f"batch_weights: {batch_weights}")
            # print(f"batch_y: {batch_y}")
            # print(f"batch_X: {batch_X}")
            with tf.GradientTape() as tape:
                predictions = model(batch_X, training=training)
                loss = loss_fn(batch_y, predictions, sample_weights=label_weights_dict)

            if training:
                gradients = tape.gradient(loss, model.trainable_variables)
                # print(f"Gradients: {gradients}")
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            epoch_loss += loss.numpy()
            num_batches += 1

            print(f"batch: {num_batches}/{len(X) // batch_size}")

        return epoch_loss / num_batches

    def train_for_one_epoch_mh(
            self,
            model: tf.keras.Model,
            optimizer: tf.keras.optimizers.Optimizer,
            primary_loss_fn,
            X: np.ndarray,
            y: np.ndarray,
            batch_size: int,
            gamma_coeff: Optional[float] = None,
            lambda_coeff: Optional[float] = None,
            sample_weights: Optional[np.ndarray] = None,
            joint_weights: Optional[np.ndarray] = None,
            joint_weight_indices: Optional[List[Tuple[int, int]]] = None,
            with_reg=False,
            with_ae=False,
            training: bool = True) -> float:
        """
        Train the model for one epoch.
        processing the batches with indices is what making it slow
        :param with_ae:
        :param with_reg:
        :param model: The model to train.
        :param optimizer: The optimizer to use.
        :param primary_loss_fn: The primary loss function to use.
        :param X: The feature set.
        :param y: The labels.
        :param batch_size: The batch size for training.
        :param gamma_coeff: Coefficient for the regressor loss.
        :param lambda_coeff: Coefficient for the decoder loss.
        :param sample_weights: Individual sample weights.
        :param joint_weights: Optional array containing all joint weights for the dataset.
        :param joint_weight_indices: Optional list of tuples, each containing a pair of indices for which a joint weight exists.
        :param training: Whether to apply training or evaluation (default is True for training).
        :return: The average loss for the epoch.
        """

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx in range(0, len(X), batch_size):
            batch_X = X[batch_idx:batch_idx + batch_size]
            batch_y = y[batch_idx:batch_idx + batch_size]
            batch_sample_weights = None if sample_weights is None \
                else sample_weights[batch_idx:batch_idx + batch_size]

            if len(batch_y) <= 1:
                # can't form a pair so skip
                continue

            # Get the corresponding joint weights for this batch
            batch_weights = None
            if joint_weights is not None and joint_weight_indices is not None:
                batch_weights = self.process_batch_weights(
                    np.arange(batch_idx, batch_idx + batch_size), joint_weights, joint_weight_indices)

            with tf.GradientTape() as tape:
                outputs = model(batch_X, training=training)

                # Unpack the outputs based on the model configuration
                if with_reg and with_ae:
                    primary_predictions, regressor_predictions, decoder_predictions = outputs
                elif with_reg:
                    primary_predictions, regressor_predictions = outputs
                    decoder_predictions = None
                elif with_ae:
                    primary_predictions, decoder_predictions = outputs
                    regressor_predictions = None
                else:
                    primary_predictions = outputs
                    regressor_predictions, decoder_predictions = None, None

                # Primary loss
                primary_loss = primary_loss_fn(batch_y, primary_predictions, sample_weights=batch_weights)

                # Regressor loss
                regressor_loss = 0
                if with_reg and gamma_coeff is not None:
                    regressor_loss = tf.keras.losses.mean_squared_error(batch_y, regressor_predictions)
                    if batch_sample_weights is not None:
                        regressor_loss = tf.cast(regressor_loss, batch_sample_weights.dtype)
                        regressor_loss = tf.reduce_sum(regressor_loss * batch_sample_weights) / tf.reduce_sum(
                            batch_sample_weights)
                    regressor_loss *= gamma_coeff

                # Decoder loss
                decoder_loss = 0
                if with_ae and lambda_coeff is not None:
                    decoder_loss = tf.keras.losses.mean_squared_error(batch_X, decoder_predictions)
                    decoder_loss *= lambda_coeff

                # Make sure all loss tensors have the same dtype
                dtype_to_use = tf.float32  # or tf.float64 based on your preference

                primary_loss = tf.cast(primary_loss, dtype_to_use)
                regressor_loss = tf.cast(regressor_loss, dtype_to_use)
                decoder_loss = tf.cast(decoder_loss, dtype_to_use)

                # Total loss
                total_loss = primary_loss + regressor_loss + decoder_loss

            if training:
                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Make sure total_loss is reduced to a single scalar value.
            total_loss_scalar = tf.reduce_sum(total_loss)

            # Update epoch_loss
            epoch_loss += total_loss_scalar.numpy()

            num_batches += 1

            print(f"batch: {num_batches}/{len(X) // batch_size}")

        return epoch_loss / num_batches

    def train_pds_dl(self,
                     model: tf.keras.Model,
                     X_subtrain: np.ndarray,
                     y_subtrain: np.ndarray,
                     X_val: np.ndarray,
                     y_val: np.ndarray,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     subtrain_label_weights_dict: Optional[Dict[float, float]] = None,
                     train_label_weights_dict: Optional[Dict[float, float]] = None,
                     learning_rate: float = 1e-3,
                     epochs: int = 100,
                     batch_size: int = 32,
                     patience: int = 9,
                     save_tag: Optional[str] = None,
                     callbacks_list=None,
                     verbose: int = 1) -> dict:
        """
        Custom training loop to train the model and returns the training history.

        :param X_train: training and validation sets together
        :param y_train: labels of training and validation sets together
        :param model: The TensorFlow model to train.
        :param X_subtrain: The training feature set.
        :param y_subtrain: The training labels.
        :param X_val: Validation features.
        :param y_val: Validation labels.
        :param subtrain_label_weights_dict: Dictionary containing label weights for the subtrain set.
        :param train_label_weights_dict: Dictionary containing label weights for the train set.
        :param learning_rate: The learning rate for the Adam optimizer.
        :param epochs: The maximum number of epochs for training.
        :param batch_size: The batch size for training.
        :param patience: The number of epochs with no improvement to wait before early stopping.
        :param save_tag: Tag to use for saving experiments.
        :param callbacks_list: List of callback instances to apply during training.
        :param verbose: Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.


        :return: The training history as a dictionary.
        """

        if callbacks_list is None:
            callbacks_list = []

        # Initialize early stopping and model checkpointing if not explicitly provided
        if not any(isinstance(cb, callbacks.EarlyStopping) for cb in callbacks_list):
            early_stopping_cb = callbacks.EarlyStopping(
                monitor='val_loss', patience=patience, restore_best_weights=True)
            callbacks_list.append(early_stopping_cb)

        if not any(isinstance(cb, callbacks.ModelCheckpoint) for cb in callbacks_list):
            checkpoint_cb = callbacks.ModelCheckpoint(f"model_weights_{str(save_tag)}.h5", save_weights_only=True)
            callbacks_list.append(checkpoint_cb)

        # Setting up callback environment
        params = {
            'epochs': epochs,
            'steps': None,
            'verbose': verbose,
            'do_validation': True,
            'metrics': ['loss', 'val_loss'],
        }
        for cb in callbacks_list:
            cb.set_model(model)
            cb.set_params(params)

        logs = {}
        # Signal the beginning of training
        for cb in callbacks_list:
            cb.on_train_begin(logs=logs)

        # Save initial weights for retraining on full training set after best epoch found
        initial_weights = model.get_weights()

        # Initialize early stopping and best epoch variables
        best_val_loss = float('inf')
        best_epoch = 0
        epochs_without_improvement = 0

        # Initialize TensorBoard
        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        #
        # print("Run the command line:\n tensorboard --logdir logs/fit")

        # Optimizer and history initialization
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        history = {'loss': [], 'val_loss': []}

        for epoch in range(epochs):
            for cb in callbacks_list:
                cb.on_epoch_begin(epoch, logs=logs)
            train_loss = self.train_for_one_epoch(
                model, optimizer, self.pds_loss_dl_vec,
                X_subtrain, y_subtrain,
                batch_size=batch_size if batch_size > 0 else len(y_subtrain),
                label_weights_dict=subtrain_label_weights_dict)

            val_loss = self.train_for_one_epoch(
                model, optimizer, self.pds_loss_dl_vec, X_val, y_val,
                batch_size=batch_size if batch_size > 0 else len(y_val),
                label_weights_dict=None, training=False)

            # Log and save epoch losses
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss}, Validation Loss: {val_loss}")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                # Save the model weights
                model.save_weights(f"best_model_weights_{str(save_tag)}.h5")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print("Early stopping triggered.")
                    break

            logs = {'loss': train_loss, 'val_loss': val_loss}  # Update logs with your training and validation loss

            for cb in callbacks_list:
                cb.on_epoch_end(epoch, logs=logs)

        for cb in callbacks_list:
            cb.on_train_end(logs=logs)

        # Plotting the losses
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.savefig(f"training_plot_{str(save_tag)}.png")
        plt.close()

        # Retraining on the combined dataset
        print(f"Retraining to the best epoch: {best_epoch}")
        # Reset history for retraining
        retrain_history = {'loss': []}

        # Reset model weights to initial state before retraining
        model.set_weights(initial_weights)

        # NOTE: test if this fixes the issue
        # Retrain up to the best epoch
        for epoch in range(best_epoch):
            retrain_loss = self.train_for_one_epoch(
                model, optimizer,
                self.pds_loss_dl_vec,
                X_train, y_train,
                batch_size=batch_size if batch_size > 0 else len(y_train),
                label_weights_dict=train_label_weights_dict)

            # Log the retrain loss
            retrain_history['loss'].append(retrain_loss)
            print(f"Retrain Epoch {epoch + 1}/{best_epoch}, Loss: {retrain_loss}")


        # Save the final model
        model.save_weights(f"final_model_weights_{str(save_tag)}.h5")
        # print where the model weights are saved
        print(f"Model weights are saved in final_model_weights_{str(save_tag)}.h5")

        return history

    # def train_pds_dl_bs(self,
    #                     model: tf.keras.Model,
    #                     X_subtrain: np.ndarray,
    #                     y_subtrain: np.ndarray,
    #                     X_val: np.ndarray,
    #                     y_val: np.ndarray,
    #                     X_train: np.ndarray,
    #                     y_train: np.ndarray,
    #                     sample_joint_weights: Optional[np.ndarray] = None,
    #                     sample_joint_weights_indices: Optional[List[Tuple[int, int]]] = None,
    #                     val_sample_joint_weights: Optional[np.ndarray] = None,
    #                     val_sample_joint_weights_indices: Optional[List[Tuple[int, int]]] = None,
    #                     train_sample_joint_weights: Optional[np.ndarray] = None,
    #                     train_sample_joint_weights_indices: Optional[List[Tuple[int, int]]] = None,
    #                     learning_rate: float = 1e-3,
    #                     epochs: int = 100,
    #                     batch_sizes=None,
    #                     patience: int = 9,
    #                     save_tag: Optional[str] = None) -> dict:
    #     """
    #     Custom training loop to train the model and returns the training history.
    #     Per epoch batch size variation
    #
    #     :param train_sample_joint_weights_indices:
    #     :param train_sample_joint_weights:
    #     :param y_train:
    #     :param X_train:
    #     :param model: The TensorFlow model to train.
    #     :param X_subtrain: The training feature set.
    #     :param y_subtrain: The training labels.
    #     :param X_val: Validation features.
    #     :param y_val: Validation labels.
    #     :param sample_joint_weights: The reweighting factors for pairs of labels in training set.
    #     :param sample_joint_weights_indices: Indices of the reweighting factors in training set.
    #     :param val_sample_joint_weights: The reweighting factors for pairs of labels in validation set.
    #     :param val_sample_joint_weights_indices: Indices of the reweighting factors in validation set.
    #     :param learning_rate: The learning rate for the Adam optimizer.
    #     :param epochs: The maximum number of epochs for training.
    #     :param batch_sizes: The batch size for training.
    #     :param patience: The number of epochs with no improvement to wait before early stopping.
    #     :param save_tag: Tag to use for saving experiments.
    #     :return: The training history as a dictionary.
    #     """
    #
    #     # Initialize early stopping and best epoch variables
    #     if batch_sizes is None:
    #         batch_sizes = [32]
    #     best_val_loss = float('inf')
    #     best_epoch = 0
    #     epochs_without_improvement = 0
    #
    #     # Initialize TensorBoard
    #     # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #     # tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #     #
    #     # print("Run the command line:\n tensorboard --logdir logs/fit")
    #
    #     # Optimizer and history initialization
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #     history = {'loss': [], 'val_loss': []}
    #
    #     for epoch in range(epochs):
    #         batch_size = random.choice(batch_sizes)
    #         train_loss = self.train_for_one_epoch(
    #             model, optimizer,
    #             self.pds_loss_dl_vec,
    #             X_subtrain, y_subtrain,
    #             batch_size=batch_size if batch_size > 0 else len(y_subtrain),
    #             joint_weights=sample_joint_weights,
    #             joint_weight_indices=sample_joint_weights_indices)
    #
    #         val_loss = self.train_for_one_epoch(
    #             model, optimizer,
    #             self.pds_loss_dl_vec,
    #             X_val, y_val,
    #             batch_size=batch_size if batch_size > 0 else len(y_val),
    #             training=False,
    #             joint_weights=val_sample_joint_weights,
    #             joint_weight_indices=val_sample_joint_weights_indices)
    #
    #         # Log and save epoch losses
    #         history['loss'].append(train_loss)
    #         history['val_loss'].append(val_loss)
    #
    #         print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss}, Validation Loss: {val_loss}")
    #
    #         # Early stopping logic
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             best_epoch = epoch
    #             epochs_without_improvement = 0
    #             # Save the model weights
    #             model.save_weights(f"best_model_weights_{str(save_tag)}.h5")
    #         else:
    #             epochs_without_improvement += 1
    #             if epochs_without_improvement >= patience:
    #                 print("Early stopping triggered.")
    #                 break
    #
    #     # Plotting the losses
    #     plt.plot(history['loss'], label='Training Loss')
    #     plt.plot(history['val_loss'], label='Validation Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.title('Training and Validation Loss Over Epochs')
    #     plt.legend()
    #     plt.savefig(f"training_plot_{str(save_tag)}.png")
    #     plt.close()
    #
    #     # Retraining on the combined dataset
    #     print(f"Retraining to the best epoch: {best_epoch}")
    #     # Reset history for retraining
    #     retrain_history = {'loss': []}
    #
    #     # NOTE: test if this fixes the issue
    #     # Retrain up to the best epoch
    #     for epoch in range(best_epoch):
    #         batch_size = random.choice(batch_sizes)
    #         retrain_loss = self.train_for_one_epoch(
    #             model, optimizer,
    #             self.pds_loss_dl_vec,
    #             X_train, y_train,
    #             batch_size=batch_size if batch_size > 0 else len(y_train),
    #             joint_weights=train_sample_joint_weights,
    #             joint_weight_indices=train_sample_joint_weights_indices)
    #
    #         # Log the retrain loss
    #         retrain_history['loss'].append(retrain_loss)
    #
    #         print(f"Retrain Epoch {epoch + 1}/{best_epoch}, Loss: {retrain_loss}")
    #
    #     # Save the final model
    #     model.save_weights(f"final_model_weights_{str(save_tag)}.h5")
    #
    #     return history

    def train_pds_dl_heads(self,
                           model: tf.keras.Model,
                           X_subtrain: np.ndarray,
                           y_subtrain: np.ndarray,
                           X_val: np.ndarray,
                           y_val: np.ndarray,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           sample_joint_weights: Optional[np.ndarray] = None,
                           sample_joint_weights_indices: Optional[List[Tuple[int, int]]] = None,
                           val_sample_joint_weights: Optional[np.ndarray] = None,
                           val_sample_joint_weights_indices: Optional[List[Tuple[int, int]]] = None,
                           train_sample_joint_weights: Optional[np.ndarray] = None,
                           train_sample_joint_weights_indices: Optional[List[Tuple[int, int]]] = None,
                           sample_weights: Optional[np.ndarray] = None,
                           val_sample_weights: Optional[np.ndarray] = None,
                           train_sample_weights: Optional[np.ndarray] = None,
                           with_reg: bool = False,
                           with_ae: bool = False,
                           learning_rate: float = 1e-3,
                           epochs: int = 100,
                           batch_size: int = 32,
                           patience: int = 9,
                           save_tag: Optional[str] = None) -> dict:
        """
        Custom training loop to train the model and returns the training history.

        :param y_train:
        :param X_train:
        :param train_sample_joint_weights:
        :param train_sample_joint_weights_indices:
        :param train_sample_weights:
        :param with_ae:
        :param with_reg:
        :param sample_weights:
        :param val_sample_weights:
        :param model: The TensorFlow model to train.
        :param X_subtrain: The training feature set.
        :param y_subtrain: The training labels.
        :param X_val: Validation features.
        :param y_val: Validation labels.
        :param sample_joint_weights: The reweighting factors for pairs of labels in training set.
        :param sample_joint_weights_indices: Indices of the reweighting factors in training set.
        :param val_sample_joint_weights: The reweighting factors for pairs of labels in validation set.
        :param val_sample_joint_weights_indices: Indices of the reweighting factors in validation set.
        :param learning_rate: The learning rate for the Adam optimizer.
        :param epochs: The maximum number of epochs for training.
        :param batch_size: The batch size for training.
        :param patience: The number of epochs with no improvement to wait before early stopping.
        :param save_tag: Tag to use for saving experiments.
        :return: The training history as a dictionary.
        """

        # Initialize early stopping and best epoch variables
        best_val_loss = float('inf')
        best_epoch = 0
        epochs_without_improvement = 0
        epochs_for_estimation = 5

        gamma_coeff, lambda_coeff = self.estimate_gamma_lambda_coeffs(
            model, X_subtrain, y_subtrain, self.pds_loss_dl_vec,
            sample_weights, sample_joint_weights, sample_joint_weights_indices,
            learning_rate=learning_rate, n_epochs=epochs_for_estimation,
            batch_size=batch_size if batch_size > 0 else len(y_subtrain),
            with_ae=with_ae, with_reg=with_reg)

        print(f'found gamma: {gamma_coeff}, lambda: {lambda_coeff}')

        # Initialize TensorBoard
        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        #
        # print("Run the command line:\n tensorboard --logdir logs/fit")

        # Optimizer and history initialization
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        history = {'loss': [], 'val_loss': []}

        for epoch in range(epochs):
            train_loss = self.train_for_one_epoch_mh(
                model, optimizer, self.pds_loss_dl_vec, X_subtrain, y_subtrain,
                batch_size=batch_size if batch_size > 0 else len(y_subtrain)
                , gamma_coeff=gamma_coeff, lambda_coeff=lambda_coeff,
                sample_weights=sample_weights, joint_weights=sample_joint_weights,
                joint_weight_indices=sample_joint_weights_indices, with_reg=with_reg, with_ae=with_ae)

            val_loss = self.train_for_one_epoch_mh(
                model, optimizer, self.pds_loss_dl_vec, X_val, y_val,
                batch_size=batch_size if batch_size > 0 else len(y_val),
                gamma_coeff=gamma_coeff, lambda_coeff=lambda_coeff,
                sample_weights=val_sample_weights, joint_weights=val_sample_joint_weights,
                joint_weight_indices=val_sample_joint_weights_indices, with_reg=with_reg, with_ae=with_ae,
                training=False)

            # Log and save epoch losses
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss}, Validation Loss: {val_loss}")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                # Save the model weights
                model.save_weights(f"best_model_weights_{str(save_tag)}.h5")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print("Early stopping triggered.")
                    break

        # Plotting the losses
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.savefig(f"training_plot_{str(save_tag)}.png")
        plt.close()

        # Retraining on the combined dataset
        print(f"Retraining to the best epoch: {best_epoch}")

        # Reset history for retraining
        retrain_history = {'loss': []}

        # Retrain up to the best epoch
        for epoch in range(best_epoch):
            retrain_loss = self.train_for_one_epoch_mh(
                model, optimizer, self.pds_loss_dl_vec, X_train, y_train,
                batch_size=batch_size if batch_size > 0 else len(y_train),
                gamma_coeff=gamma_coeff, lambda_coeff=lambda_coeff,
                sample_weights=train_sample_weights,
                joint_weights=train_sample_joint_weights,
                joint_weight_indices=train_sample_joint_weights_indices,
                with_reg=with_reg, with_ae=with_ae)

            # Log the retrain loss
            retrain_history['loss'].append(retrain_loss)
            print(f"Retrain Epoch {epoch + 1}/{best_epoch}, Loss: {retrain_loss}")

        # Save the final model
        model.save_weights(f"final_model_weights_{str(save_tag)}.h5")

        return history

    def custom_data_generator(self, X, y, batch_size):
        """
        Yields batches of cme_files such that the last two samples in each batch
        have target labels above ln(10), and the remaining have labels below ln(10).
        Below-threshold samples cycle through before repeating.
        """
        above_threshold_indices = np.where(y > np.log(10))[0]
        below_threshold_indices = np.where(y <= np.log(10))[0]

        # Create an iterator that will cycle through the below_threshold_indices
        cyclic_below_threshold = cycle(below_threshold_indices)

        while True:
            # Select random above-threshold indices
            batch_indices_above = np.random.choice(above_threshold_indices, 2, replace=False)

            # Select (batch_size - 2) below-threshold indices in a cyclic manner
            batch_indices_below = [next(cyclic_below_threshold) for _ in range(batch_size - 2)]

            batch_indices = np.concatenate([batch_indices_below, batch_indices_above])

            batch_X = X[batch_indices]
            batch_y = y[batch_indices]

            # Shuffle the entire batch
            indices = np.arange(batch_X.shape[0])
            np.random.shuffle(indices)
            batch_X = batch_X[indices]
            batch_y = batch_y[indices]

            yield batch_X, batch_y

    def train_pds_injection(self,
                            model: Model,
                            X_subtrain: Tensor,
                            y_subtrain: Tensor,
                            X_val: Tensor,
                            y_val: Tensor,
                            X_train: Tensor,
                            y_train: Tensor,
                            learning_rate: float = 1e-3,
                            epochs: int = 100,
                            batch_size: int = 32,
                            patience: int = 9) -> callbacks.History:
        """
        Trains the model and returns the training history. injection of rare examples

        :param y_train:
        :param X_train:
        :param model: The TensorFlow model to train.
        :param X_subtrain: The training feature set.
        :param y_subtrain: The training labels.
        :param X_val: Validation features.
        :param y_val: Validation labels.
        :param learning_rate: The learning rate for the Adam optimizer.
        :param epochs: The maximum number of epochs for training.
        :param batch_size: The batch size for training.
        :param patience: The number of epochs with no improvement to wait before early stopping.
        :return: The training history as a History object.
        """

        # Create custom cme_files generators for training and validation
        train_gen = self.custom_data_generator(X_subtrain, y_subtrain, batch_size)
        val_gen = self.custom_data_generator(X_val, y_val, batch_size)

        train_steps = len(y_subtrain) // batch_size
        val_steps = len(y_val) // batch_size if len(y_val) > batch_size else len(y_val)

        # Setup TensorBoard
        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        #
        # print("Run the command line:\n tensorboard --logdir logs/fit")

        # Setup early stopping
        early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # checkpoint callback
        # Setup model checkpointing
        checkpoint_cb = callbacks.ModelCheckpoint("model_weights.h5", save_weights_only=True)
        # Create an instance of the custom callback

        # Include weighted_loss_cb in callbacks only if sample_joint_weights is not None
        callback_list = [early_stopping_cb, checkpoint_cb]

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=self.pds_loss_vec)

        # First train the model with a validation set to determine the best epoch
        history = model.fit(train_gen,
                            steps_per_epoch=train_steps,
                            validation_data=val_gen,
                            validation_steps=val_steps,
                            epochs=epochs,
                            batch_size=batch_size if batch_size > 0 else len(y_subtrain),
                            validation_batch_size=batch_size if batch_size > 0 else len(y_val),
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

        # Create custom generators for combined cme_files
        train_gen_comb = self.custom_data_generator(X_train, y_train, batch_size)

        # Calculate the number of steps per epoch for training
        train_steps_comb = len(X_train) // batch_size

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=self.pds_loss_vec)

        model.fit(train_gen_comb,
                  steps_per_epoch=train_steps_comb,
                  epochs=best_epoch,
                  batch_size=batch_size if batch_size > 0 else len(y_train),
                  callbacks=[checkpoint_cb])

        return history

    # def train_pds_fast(self,
    #                         model: Model,
    #                         X_subtrain: Tensor,
    #                         y_subtrain: Tensor,
    #                         X_val: Tensor,
    #                         y_val: Tensor,
    #                         sample_joint_weights: ndarray = None,
    #                         learning_rate: float = 1e-3,
    #                         epochs: int = 100,
    #                         batch_size: int = 32,
    #                         patience: int = 9) -> callbacks.History:
    #     """
    #     Trains the model and returns the training history.
    #     TODO: fix this issue where loss values are not correct
    #     :param model: The TensorFlow model to train.
    #     :param X_subtrain: The training feature set.
    #     :param y_subtrain: The training labels.
    #     :param X_val: Validation features.
    #     :param y_val: Validation labels.
    #     :param sample_joint_weights: The reweighting factors for pairs of labels.
    #     :param learning_rate: The learning rate for the Adam optimizer.
    #     :param epochs: The maximum number of epochs for training.
    #     :param batch_size: The batch size for training.
    #     :param patience: The number of epochs with no improvement to wait before early stopping.
    #     :return: The training history as a History object.
    #     """
    #
    #     # Setup TensorBoard
    #     log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #     tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #
    #     print("Run the command line:\n tensorboard --logdir logs/fit")
    #
    #     # Setup early stopping
    #     early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    #
    #     # In your Callback
    #     class WeightedLossCallback(callbacks.Callback):
    #         def on_subtrain_batch_begin(self, batch, logs=None):
    #             idx1, idx2 = np.triu_indices(len(y_subtrain), k=1)
    #             one_d_indices = [map_to_1D_idx(i, j, len(y_subtrain)) for i, j in zip(idx1, idx2)]
    #             joint_weights_batch = sample_joint_weights[one_d_indices]  # Retrieve weights for this batch
    #             self.model.loss_weights = joint_weights_batch  # Set loss weights for this batch
    #
    #     # Create an instance of the custom callback
    #     weighted_loss_cb = WeightedLossCallback()
    #
    #     # Include weighted_loss_cb in callbacks only if sample_joint_weights is not None
    #     callback_list = [ early_stopping_cb]
    #     if sample_joint_weights is not None:
    #         callback_list.append(weighted_loss_cb)
    #
    #     # Compile the model
    #     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=self.pds_loss_fast)
    #
    #     # First train the model with a validation set to determine the best epoch
    #     history = model.fit(X_subtrain, y_subtrain,
    #                         epochs=epochs,
    #                         batch_size=batch_size,
    #                         validation_data=(X_val, y_val),
    #                         callbacks=callback_list)
    #
    #     # Get the best epoch from early stopping
    #     best_epoch = np.argmin(history.history['val_loss']) + 1
    #
    #     # Plot training loss and validation loss
    #     plt.plot(history.history['loss'], label='Training Loss')
    #     plt.plot(history.history['val_loss'], label='Validation Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.title('Training and Validation Loss Over Epochs')
    #     plt.legend()
    #     plt.show()
    #
    #     # Retrain the model on the combined dataset (training + validation) to the best epoch found
    #     X_combined = np.concatenate((X_subtrain, X_val), axis=0)
    #     y_combined = np.concatenate((y_subtrain, y_val), axis=0)
    #
    #     if sample_joint_weights is not None:
    #         sample_joint_weights_combined = np.concatenate((sample_joint_weights, sample_joint_weights), axis=0)
    #
    #     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=self.pds_loss_fast)
    #     if sample_joint_weights is not None:
    #         model.fit(X_combined, y_combined, epochs=best_epoch, batch_size=batch_size, callbacks=[weighted_loss_cb])
    #     else:
    #         model.fit(X_combined, y_combined, epochs=best_epoch, batch_size=batch_size)
    #
    #     return history

    def train_reg_head(self,
                       model: Model,
                       X_subtrain: ndarray,
                       y_subtrain: ndarray,
                       X_val: ndarray,
                       y_val: ndarray,
                       X_train: ndarray,
                       y_train: ndarray,
                       sample_weights: Optional[ndarray] = None,
                       sample_val_weights: Optional[ndarray] = None,
                       sample_train_weights: Optional[ndarray] = None,
                       learning_rate: float = 1e-3,
                       epochs: int = 100,
                       batch_size: int = 32,
                       patience: int = 9,
                       save_tag=None) -> callbacks.History:
        """
        Train a neural network model focusing only on the regression output.
        Include reweighting for balancing the loss.

        :param save_tag:
        :param model: The neural network model.
        :param X_subtrain: sub Training features.
        :param y_subtrain: sub Training labels.
        :param X_val: Validation features.
        :param y_val: Validation labels.
        :param X_train: Training features.
        :param y_train: Training labels.
        :param sample_weights: Sample weights for sub training set.
        :param sample_val_weights: Sample weights for validation set.
        :param sample_train_weights: Sample weights for training
        :param learning_rate: Learning rate for Adam optimizer.
        :param epochs: Number of epochs.
        :param batch_size: Batch size.
        :param patience: Number of epochs for early stopping.
        :return: Training history.
        """

        # Setup TensorBoard
        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        #
        # print("Run the command line:\n tensorboard --logdir logs/fit")

        # Early stopping callback
        early_stopping_cb = callbacks.EarlyStopping(monitor='val_regression_head_loss', patience=patience,
                                                    restore_best_weights=True)
        # Setup model checkpointing
        checkpoint_cb = callbacks.ModelCheckpoint(f"model_weights_{str(save_tag)}.h5", save_weights_only=True)
        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss={'regression_head': 'mse'})

        # Train the model with a validation set
        history = model.fit(X_subtrain, {'regression_head': y_subtrain},
                            sample_weight=sample_weights,
                            epochs=epochs,
                            batch_size=batch_size if batch_size > 0 else len(y_subtrain),
                            validation_data=(X_val, {'regression_head': y_val}, sample_val_weights),
                            validation_batch_size=batch_size if batch_size > 0 else len(y_val),
                            callbacks=[early_stopping_cb, checkpoint_cb])

        # Find the best epoch from early stopping
        best_epoch = np.argmin(history.history['val_regression_head_loss']) + 1

        # Plot training and validation loss
        plt.plot(history.history['regression_head_loss'], label='Training Loss')
        plt.plot(history.history['val_regression_head_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        file_path = f"training_reg_plot_{str(save_tag)}.png"
        plt.savefig(file_path)
        plt.close()

        # Retrain the model to the best epoch using combined cme_files
        model.fit(X_train, {'regression_head': y_train},
                  sample_weight=sample_train_weights,
                  epochs=best_epoch,
                  batch_size=batch_size if batch_size > 0 else len(y_train),
                  callbacks=[checkpoint_cb])

        # save the model weights
        model.save_weights(f"extended_model_weights_{str(save_tag)}.h5")

        return history

    def estimate_gamma_lambda_coeffs(self,
                                     model: tf.keras.Model,
                                     X_subtrain: np.ndarray,
                                     y_subtrain: np.ndarray,
                                     primary_loss_fn,
                                     sample_weights: Optional[np.ndarray] = None,
                                     sample_joint_weights: Optional[np.ndarray] = None,
                                     sample_joint_weights_indices: Optional[List[Tuple[int, int]]] = None,
                                     learning_rate: float = 1e-3, n_epochs: int = 10,
                                     batch_size: int = 32,
                                     with_ae=False, with_reg=False) -> Tuple[float, float]:
        """
        Estimate the gamma and lambda coefficients for balancing the primary, regression, and decoder losses.

        :param with_ae:
        :param with_reg:
        :param sample_joint_weights:
        :param sample_joint_weights_indices:
        :param model: The neural network model.
        :param X_subtrain: Training features.
        :param y_subtrain: Training labels.
        :param primary_loss_fn: Primary loss function.
        :param sample_weights: Sample weights for training set.
        :param sample_val_weights: Sample weights for validation set.
        :param learning_rate: Learning rate for Adam optimizer.
        :param n_epochs: Number of epochs to train each branch for coefficient estimation.
        :param batch_size: Batch size.
        :return: Estimated gamma and lambda coefficients.
        """

        # Initialize lists to store validation losses for each head
        primary_losses = []
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # Train the primary head using custom training loop
        for epoch in range(n_epochs):
            train_loss = self.train_for_one_epoch_mh(
                model, optimizer, primary_loss_fn,
                X_subtrain, y_subtrain,
                batch_size,
                sample_weights=sample_weights,
                joint_weights=sample_joint_weights,
                joint_weight_indices=sample_joint_weights_indices,
                training=True,
                with_ae=with_ae, with_reg=with_reg)
            primary_losses.append(train_loss)

        reg_losses = []
        dec_losses = []

        # Train regression branch only if with_reg is True
        if with_reg:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss={'regression_head': 'mse'})
            history_reg = model.fit(X_subtrain, {'regression_head': y_subtrain},
                                    sample_weight=sample_weights,
                                    epochs=n_epochs,
                                    batch_size=batch_size)
            reg_losses = history_reg.history['loss']

        # Train decoder branch only if with_ae is True
        if with_ae:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss={'decoder_head': 'mse'})
            history_dec = model.fit(X_subtrain, {'decoder_head': X_subtrain},
                                    sample_weight=sample_weights,
                                    epochs=n_epochs,
                                    batch_size=batch_size)
            dec_losses = history_dec.history['loss']

        # # Initialize coefficients to None
        # gamma_coef = None
        # lambda_coef = None
        #
        # # Calculate gamma and lambda as the sum of the ratios, if applicable
        # if with_reg:
        #     gamma_ratios = [p / r for p, r in zip(primary_losses, reg_losses)]
        #     gamma_coef = np.mean(gamma_ratios)
        #
        # if with_ae:
        #     lambda_ratios = [p / d for p, d in zip(primary_losses, dec_losses)]
        #     lambda_coef = np.mean(lambda_ratios)

        # Initialize coefficients to None
        gamma_coef = None
        lambda_coef = None

        # Assuming primary_losses, reg_losses, and dec_losses are lists or could be converted to NumPy arrays
        primary_losses = np.array(primary_losses)

        # Calculate gamma and lambda as the mean of the ratios, if applicable
        if with_reg:
            reg_losses = np.array(reg_losses)
            gamma_ratios = primary_losses / reg_losses
            gamma_coef = np.mean(gamma_ratios)

        if with_ae:
            dec_losses = np.array(dec_losses)
            lambda_ratios = primary_losses / dec_losses
            lambda_coef = np.mean(lambda_ratios)

        return gamma_coef, lambda_coef

    def estimate_lambda_coef(self,
                             model,
                             X_subtrain, y_subtrain,
                             sample_weights=None,
                             learning_rate=1e-3,
                             n_epochs=10,
                             batch_size=32):
        """
        Estimate the lambda coefficient for balancing the regression and decoder losses.

        :param model: The neural network model.
        :param X_subtrain: Training features.
        :param y_subtrain: Training labels.
        :param sample_weights: Sample weights for training set.
        :param learning_rate: Learning rate for Adam optimizer.
        :param n_epochs: Number of epochs to train each branch for lambda estimation.
        :param batch_size: Batch size.
        :return: Estimated lambda coefficient.
        """

        # Train regression branch only
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss={'regression_head': 'mse'})
        history_reg = model.fit(X_subtrain, {'regression_head': y_subtrain},
                                sample_weight=sample_weights,
                                epochs=n_epochs,
                                batch_size=batch_size)

        reg_losses = history_reg.history['loss']

        # Train decoder branch only
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss={'decoder_head': 'mse'})
        history_dec = model.fit(X_subtrain, {'decoder_head': X_subtrain},
                                sample_weight=sample_weights,
                                epochs=n_epochs,
                                batch_size=batch_size)

        dec_losses = history_dec.history['loss']

        # Calculate lambda as the sum of the ratios
        # ratios = [r / d for r, d in zip(reg_losses, dec_losses)]
        # lambda_coef = np.mean(ratios)
        reg_losses = np.array(reg_losses)
        dec_losses = np.array(dec_losses)

        # Compute ratios using element-wise division
        ratios = reg_losses / dec_losses

        # Calculate the mean of the ratios
        lambda_coef = np.mean(ratios)

        return lambda_coef

    def train_reg_ae_heads(self, model: Model,
                           X_subtrain: ndarray,
                           y_subtrain: ndarray,
                           X_val: ndarray,
                           y_val: ndarray,
                           X_train: ndarray,
                           y_train: ndarray,
                           sample_weights: Optional[ndarray] = None,
                           sample_val_weights: Optional[ndarray] = None,
                           sample_train_weights: Optional[ndarray] = None,
                           learning_rate: float = 1e-3,
                           epochs: int = 100,
                           batch_size: int = 32,
                           patience: int = 9,
                           save_tag=None) -> callbacks.History:
        """
        Train a neural network model focusing on the regression and autoencoder output.
        Includes reweighting for balancing the loss and saves the model weights.

        :param model: The neural network model.
        :param X_subtrain: Training features.
        :param y_subtrain: Training labels.
        :param X_val: Validation features.
        :param y_val: Validation labels.
        :param sample_weights: Sample weights for training set.
        :param sample_val_weights: Sample weights for validation set.
        :param learning_rate: Learning rate for Adam optimizer.
        :param epochs: Number of epochs.
        :param batch_size: Batch size.
        :param patience: Number of epochs for early stopping.
        :param save_tag: Tag for saving model weights and plots.
        :return: Training history.
        """

        epochs_for_estimation = 5

        lambda_coef = self.estimate_lambda_coef(model, X_subtrain, y_subtrain,
                                                sample_weights,
                                                learning_rate, epochs_for_estimation,
                                                batch_size=batch_size if batch_size > 0 else len(y_subtrain))

        print(f"Lambda coefficient found: {lambda_coef}")

        # Setup TensorBoard
        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # print("Run the command line:\n tensorboard --logdir logs/fit")

        # Early stopping callback
        early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Model checkpointing
        checkpoint_cb = callbacks.ModelCheckpoint(f"model_weights_ae_{str(save_tag)}.h5", save_weights_only=True)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss={'regression_head': 'mse', 'decoder_head': 'mse'},
                      loss_weights={'regression_head': 1.0, 'decoder_head': lambda_coef})

        # Prepare cme_files dictionary
        y_dict = {'regression_head': y_subtrain, 'decoder_head': X_subtrain}
        val_y_dict = {'regression_head': y_val, 'decoder_head': X_val}

        # Train the model
        history = model.fit(X_subtrain, y_dict,
                            sample_weight=sample_weights,
                            epochs=epochs,
                            batch_size=batch_size if batch_size > 0 else len(y_subtrain),
                            validation_data=(X_val, val_y_dict, sample_val_weights),
                            validation_batch_size=batch_size if batch_size > 0 else len(y_val),
                            callbacks=[early_stopping_cb, checkpoint_cb])

        # Find the best epoch from early stopping
        best_epoch = np.argmin(history.history['val_loss']) + 1

        # Plot training and validation loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        file_path = f"training_ae_plot_{str(save_tag)}.png"
        plt.savefig(file_path)
        plt.close()

        # Retrain the model to the best epoch using combined cme_files
        model.fit(X_train, {'regression_head': y_train, 'decoder_head': X_train},
                  sample_weight=sample_train_weights,
                  epochs=best_epoch,
                  batch_size=batch_size if batch_size > 0 else len(y_train),
                  callbacks=[checkpoint_cb])

        # Save the extended model weights
        model.save_weights(f"extended_model_weights_ae_{str(save_tag)}.h5")

        return history

    def plot_model(self, model: Model, name: str) -> None:
        """
        Plot the model architecture and save the figure.
        :param name: name of the file
        :param model: The model to plot.
        :return: None
        """
        tf.keras.utils.plot_model(model, to_file=f'./{name}.png', show_shapes=True, show_layer_names=True)

    def error_vectorized(self, z1: tf.Tensor, z2: tf.Tensor, label1: tf.Tensor, label2: tf.Tensor) -> tf.Tensor:
        """
        Vectorized function to compute the error between the zdist of two batches of input predicted z values
        and their ydist. Range of the error is [0, 8].

        :param z1: A tensor containing the predicted z values for the first batch of input samples.
        :param z2: A tensor containing the predicted z values for the second batch of input samples.
        :param label1: A tensor containing the labels of the first batch of input samples.
        :param label2: A tensor containing the labels of the second batch of input samples.
        :return: A tensor containing the squared differences between the zdist and ydist for each pair.
        """
        z_distance = tf.reduce_sum(tf.square(z1 - z2), axis=-1)
        y_distance = tf.square(label1 - label2)
        squared_difference = 0.5 * tf.square(z_distance - y_distance)

        return squared_difference

    def pds_loss_dl(self, y_true, z_pred, sample_weights=None, reduction=tf.keras.losses.Reduction.NONE):
        """
        Computes the weighted loss for a batch of predicted features and their labels.

        :param y_true: A batch of true label values, shape of [batch_size, 1].
        :param z_pred: A batch of predicted Z values, shape of [batch_size, 2].
        :param sample_weights: A batch of sample weights, shape of [batch_size, 1].
        :param reduction: The type of reduction to apply to the loss.
        :return: The weighted average error for all unique combinations of the samples in the batch.
        """
        int_batch_size = tf.shape(z_pred)[0]
        batch_size = tf.cast(int_batch_size, dtype=tf.float32)
        total_error = tf.constant(0.0, dtype=tf.float32)

        # Initialize counter for sample_weights
        weight_idx = 0

        # Loop through all unique pairs of samples in the batch
        for i in tf.range(int_batch_size):
            for j in tf.range(i + 1, int_batch_size):
                z1, z2 = z_pred[i], z_pred[j]
                label1, label2 = y_true[i], y_true[j]
                err = error(z1, z2, label1, label2)  # Assuming `error` is defined elsewhere in your code

                # Apply sample weights if provided
                if sample_weights is not None:
                    weight = sample_weights[weight_idx]  # Get the weight for this pair
                    weighted_err = err * weight
                    weight_idx += 1  # Move to the next weight
                else:
                    weighted_err = err

                total_error += tf.cast(weighted_err, dtype=tf.float32)

        if reduction == tf.keras.losses.Reduction.SUM:
            return total_error  # Total loss
        elif reduction == tf.keras.losses.Reduction.NONE:
            denom = tf.cast(batch_size * (batch_size - 1) / 2 + 1e-9, dtype=tf.float32)
            return total_error / denom  # Average loss
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}.")

    def pds_loss_dl_vec(self, y_true, z_pred, sample_weights=None, reduction=tf.keras.losses.Reduction.NONE):
        """
        Computes the weighted loss for a batch of predicted features and their labels.

        :param y_true: A batch of true label values, shape of [batch_size, 1].
        :param z_pred: A batch of predicted Z values, shape of [batch_size, 2].
        :param sample_weights: A dictionary mapping label values to their corresponding reweight.
        :param reduction: The type of reduction to apply to the loss.
        :return: The weighted average error for all unique combinations of the samples in the batch.
        """

        batch_size = tf.shape(y_true)[0]
        # Compute pairwise differences for z_pred and y_true using broadcasting
        y_true_diff = y_true - tf.transpose(y_true)
        z_pred_diff = z_pred[:, tf.newaxis, :] - z_pred[tf.newaxis, :, :]
        # Calculate squared L2 norm for z_pred differences
        z_diff_squared = tf.reduce_sum(tf.square(z_pred_diff), axis=-1)
        # Calculate squared differences for y_true
        y_diff_squared = tf.square(y_true_diff)
        # Cast y_diff_squared to match the data type of z_diff_squared
        y_diff_squared = tf.cast(y_diff_squared, dtype=z_diff_squared.dtype)
        # Compute the loss for each pair
        pairwise_loss = 0.5 * tf.square(z_diff_squared - y_diff_squared)

        # Apply sample weights if provided
        if sample_weights is not None:
            # Convert sample_weights to a lookup table
            keys = tf.constant(list(sample_weights.keys()), dtype=tf.float32)
            values = tf.constant(list(sample_weights.values()), dtype=tf.float32)
            table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values), default_value=1.0)
            # Lookup the weights for each y_true value
            weights = table.lookup(tf.reshape(y_true, [-1]))
            weights_matrix = weights[:, None] * weights[None, :]
            # Apply the weights to the pairwise loss
            pairwise_loss *= weights_matrix

        # Mask to exclude self-comparisons (where i == j)
        mask = 1 - tf.eye(batch_size, dtype=z_diff_squared.dtype)
        # Apply mask to exclude self-comparisons from the loss calculation
        pairwise_loss_masked = pairwise_loss * mask
        # Sum over all unique pairs
        total_error = tf.reduce_sum(pairwise_loss_masked)
        # Number of unique comparisons, excluding self-pairs
        num_comparisons = tf.cast(batch_size * (batch_size - 1), dtype=z_diff_squared.dtype)

        if reduction == tf.keras.losses.Reduction.SUM:
            return total_error / 2
        elif reduction == tf.keras.losses.Reduction.NONE:
            # Avoid division by zero
            return total_error / (num_comparisons + 1e-9)
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}.")

    # def expand_sample_weights_to_matrix(self, sample_weights, batch_size):
    #     """
    #     Expands a flat array of sample weights for unique pairs into a symmetric matrix.
    #
    #     :param sample_weights: A 1D tensor of sample weights for unique pairs.
    #     :param batch_size: The number of samples in the batch.
    #     :return weights_matrix: A 2D tensor of sample weights for all pairs in the batch.
    #     """
    #     # Ensure sample_weights is a 1D tensor
    #     sample_weights = tf.reshape(sample_weights, [-1])
    #
    #     # Generate indices for the rows and columns
    #     rows, cols = tf.meshgrid(tf.range(batch_size), tf.range(batch_size), indexing='ij')
    #
    #     # Identify the indices of the upper triangle (excluding the diagonal)
    #     upper_tri_indices = tf.where(rows < cols)
    #
    #     # Create a matrix of zeros with the same shape as the desired output
    #     weights_matrix = tf.zeros((batch_size, batch_size), dtype=sample_weights.dtype)
    #
    #     # Update the weights_matrix with sample_weights at the upper triangle indices
    #     weights_matrix = tf.tensor_scatter_nd_update(weights_matrix, upper_tri_indices, sample_weights)
    #
    #     # Since the weights for the pairs are symmetric, we can add the transpose to fill in the lower triangle
    #     weights_matrix += tf.transpose(weights_matrix)
    #
    #     return weights_matrix

    def update_pair_counts(self, label1, label2):
        label1 = label1[0]
        label2 = label2[0]

        is_sep_1 = label1 > np.log(10.0)
        is_sep_2 = label2 > np.log(10.0)
        is_elevated_1 = (label1 > np.log(10.0 / np.exp(2))) & (label1 <= np.log(10.0))
        is_elevated_2 = (label2 > np.log(10.0 / np.exp(2))) & (label2 <= np.log(10.0))
        is_background_1 = label1 <= np.log(10.0 / np.exp(2))
        is_background_2 = label2 <= np.log(10.0 / np.exp(2))

        self.sep_sep_count.assign_add(tf.cast(is_sep_1 & is_sep_2, tf.int32))
        self.sep_elevated_count.assign_add(tf.cast((is_sep_1 & is_elevated_2) | (is_elevated_1 & is_sep_2), tf.int32))
        self.sep_background_count.assign_add(
            tf.cast((is_sep_1 & is_background_2) | (is_background_1 & is_sep_2), tf.int32))
        self.elevated_elevated_count.assign_add(tf.cast(is_elevated_1 & is_elevated_2, tf.int32))
        self.elevated_background_count.assign_add(
            tf.cast((is_elevated_1 & is_background_2) | (is_background_1 & is_elevated_2), tf.int32))
        self.background_background_count.assign_add(tf.cast(is_background_1 & is_background_2, tf.int32))

    def pds_loss(self, y_true, z_pred, reduction=tf.keras.losses.Reduction.NONE):
        """
        Computes the loss for a batch of predicted features and their labels.
        verified!

        :param y_true: A batch of true label values, shape of [batch_size, 1].
        :param z_pred: A batch of predicted Z values, shape of [batch_size, 2].
        :param reduction: The type of reduction to apply to the loss.
        :return: The average error for all unique combinations of the samples in the batch.
        """
        int_batch_size = tf.shape(z_pred)[0]
        batch_size = tf.cast(int_batch_size, dtype=tf.float32)
        total_error = tf.constant(0.0, dtype=tf.float32)

        # tf.print(" received batch size:", int_batch_size)
        self.number_of_batches += 1

        # Loop through all unique pairs of samples in the batch
        for i in tf.range(int_batch_size):
            for j in tf.range(i + 1, int_batch_size):
                z1, z2 = z_pred[i], z_pred[j]
                # tf.print(z1, z2, sep=', ', end='\n')
                label1, label2 = y_true[i], y_true[j]
                # Update pair counts and check if it's a SEP-SEP pair
                # self.update_pair_counts(label1, label2)
                # tf.print(label1, label2, sep=', ', end='\n')
                err = error(z1, z2, label1, label2)
                # tf.print(err, end='\n\n')
                total_error += tf.cast(err, dtype=tf.float32)

                # tf.print("Pair (i, j):", i, j, "z1, z2:", z1, z2, "label1, label2:", label1, label2, "err:", err)

        # tf.print(total_error)
        # tf.print("Total error before normalization:", total_error)

        if reduction == tf.keras.losses.Reduction.SUM:
            return total_error  # total loss
        elif reduction == tf.keras.losses.Reduction.NONE:
            denom = tf.cast(batch_size * (batch_size - 1) / 2 + 1e-9, dtype=tf.float32)
            # tf.print(denom)
            return total_error / denom  # average loss
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}.")

    def pds_loss_vec(self, y_true, z_pred, reduction=tf.keras.losses.Reduction.NONE):
        """
        Vectorized computation of the loss for a batch of predicted features and their labels.

        :param y_true: A batch of true label values, shape of [batch_size, 1].
        :param z_pred: A batch of predicted Z values, shape of [batch_size, 2].
        :param reduction: The type of reduction to apply to the loss.
        :return: The average error for all unique combinations of the samples in the batch.
        """
        # Compute pairwise differences for z_pred and y_true using broadcasting
        y_true_diff = y_true - tf.transpose(y_true)
        z_pred_diff = z_pred[:, tf.newaxis, :] - z_pred[tf.newaxis, :, :]

        # Calculate squared L2 norm for z_pred differences
        z_diff_squared = tf.reduce_sum(tf.square(z_pred_diff), axis=-1)

        # Calculate squared differences for y_true
        y_diff_squared = tf.square(y_true_diff)

        # Compute the loss for each pair
        pairwise_loss = .5 * tf.square(z_diff_squared - y_diff_squared)

        # Mask to exclude self-comparisons (where i == j)
        batch_size = tf.shape(y_true)[0]
        mask = 1 - tf.eye(batch_size, dtype=tf.float32)

        # Apply mask to exclude self-comparisons from the loss calculation
        pairwise_loss_masked = pairwise_loss * mask

        # Sum over all unique pairs
        total_error = tf.reduce_sum(pairwise_loss_masked)

        # Number of unique comparisons, excluding self-pairs
        num_comparisons = tf.cast(batch_size * (batch_size - 1), dtype=tf.float32)

        if reduction == tf.keras.losses.Reduction.SUM:
            return total_error / 2
        elif reduction == tf.keras.losses.Reduction.NONE:
            # Avoid division by zero
            return total_error / (num_comparisons + 1e-9)
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


class InvestigateCallback(callbacks.Callback):
    """
    Custom callback to evaluate the model on SEP samples at the end of each epoch.
    """

    def __init__(self,
                 model: Model,
                 X_train: ndarray,
                 y_train: ndarray,
                 batch_size: int,
                 model_builder: ModelBuilder,
                 save_tag: Optional[str] = None):
        super().__init__()
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size if batch_size > 0 else None
        self.sep_threshold = np.log(10)
        self.threshold = np.log(10.0 / np.exp(2))
        self.save_tag = save_tag
        self.sep_sep_losses = []
        # self.losses = []
        # self.epochs_10s = []
        self.model_builder = model_builder
        self.sep_sep_count = 0
        self.sep_sep_counts = []
        self.cumulative_sep_sep_count = 0  # Initialize cumulative count
        self.cumulative_sep_sep_counts = []
        self.total_counts = []
        self.batch_counts = []
        self.sep_sep_percentages = []
        # the losses
        self.pair_type_losses = {  # Store losses for each pair type
            'sep_sep': [],
            'sep_elevated': [],
            'sep_background': [],
            'elevated_elevated': [],
            'elevated_background': [],
            'background_background': []
        }
        self.overall_losses = []  # Store overall losses

    def on_batch_end(self, batch, logs=None):
        """
        Actions to be taken at the end of each batch.

        :param batch: the index of the batch within the current epoch.
        :param logs: the logs containing the metrics results.
        """
        sep_indices = self.find_sep_samples(self.y_train)
        if len(sep_indices) > 0:
            X_sep = self.X_train[sep_indices]
            y_sep = self.y_train[sep_indices]
            # Evaluate the model on SEP samples
            # sep_sep_loss = self.model.evaluate(X_sep, y_sep, batch_size=len(self.y_train), verbose=0)
            sep_sep_loss = evaluate(self.model, X_sep, y_sep)
            self.sep_sep_losses.append(sep_sep_loss)

        # Add the SEP-SEP count for the current batch to the cumulative count
        batch_sep_sep_count = int(self.model_builder.sep_sep_count.numpy())
        print(f'end of batch: {batch}, sep_sep_count: {batch_sep_sep_count} in')
        self.sep_sep_count += batch_sep_sep_count
        self.cumulative_sep_sep_count += batch_sep_sep_count
        self.cumulative_sep_sep_counts.append(self.cumulative_sep_sep_count)
        # Reset for next batch
        self.model_builder.sep_sep_count.assign(0)

    def on_epoch_begin(self, epoch, logs=None):
        """
        Actions to be taken at the beginning of each epoch.

        :param epoch: the index of the epoch.
        :param logs: the logs containing the metrics results.
        """
        # Resetting the counts
        self.sep_sep_count = 0
        self.cumulative_sep_sep_count = 0
        self.model_builder.sep_sep_count.assign(0)
        self.model_builder.sep_elevated_count.assign(0)
        self.model_builder.sep_background_count.assign(0)
        self.model_builder.elevated_elevated_count.assign(0)
        self.model_builder.elevated_background_count.assign(0)
        self.model_builder.background_background_count.assign(0)
        self.model_builder.number_of_batches = 0

    def on_epoch_end(self, epoch, logs=None):
        # Find SEP samples
        # sep_indices = self.find_sep_samples(self.y_train, self.sep_threshold)
        # if len(sep_indices) > 0:
        #     X_sep = self.X_train[sep_indices]
        #     y_sep = self.y_train[sep_indices]
        #     # Evaluate the model on SEP samples
        #     sep_sep_loss = self.model.evaluate(X_sep, y_sep, verbose=0)
        #     self.sep_sep_losses.append(sep_sep_loss)
        #     print(f" Epoch {epoch + 1}: SEP-SEP Loss: {sep_sep_loss}")

        self.collect_losses(epoch)

        # Save the current counts
        self.sep_sep_counts.append(self.sep_sep_count)
        total_count = (
                self.sep_sep_count +
                int(self.model_builder.sep_elevated_count.numpy()) +
                int(self.model_builder.sep_background_count.numpy()) +
                int(self.model_builder.elevated_elevated_count.numpy()) +
                int(self.model_builder.elevated_background_count.numpy()) +
                int(self.model_builder.background_background_count.numpy())
        )
        self.total_counts.append(total_count)
        self.batch_counts.append(self.model_builder.number_of_batches)

        # Calculate and save the percentage of SEP-SEP pairs
        if total_count > 0:
            self.sep_sep_percentages.append((self.sep_sep_count / total_count) * 100)
        else:
            self.sep_sep_percentages.append(0)

        # Reset the counts for the next epoch
        # self.model_builder.sep_sep_count.assign(0)
        self.model_builder.sep_elevated_count.assign(0)
        self.model_builder.sep_background_count.assign(0)
        self.model_builder.elevated_elevated_count.assign(0)
        self.model_builder.elevated_background_count.assign(0)
        self.model_builder.background_background_count.assign(0)
        self.sep_sep_count = 0
        self.model_builder.number_of_batches = 0

        # if epoch % 10 == 9:  # every 10th epoch (considering the first epoch is 0)
        #     loss = self.model.evaluate(self.X_train, self.y_train, batch_size=len(self.y_train), verbose=0)
        #     self.losses.append(loss)
        #     self.epochs_10s.append(epoch + 1)

    def collect_losses(self, epoch):
        """
        Collects and stores the losses for each pair type and overall loss for the given epoch.

        :param epoch: Current epoch number.
        """
        # Evaluate the model and get losses for each pair type, including overall
        pair_losses = evaluate(self.model, self.X_train, self.y_train, pairs=True)

        # Store and print pair type losses
        for pair_type, loss in pair_losses.items():
            if pair_type != 'overall':  # Exclude overall loss here
                self.pair_type_losses[pair_type].append(loss)
                print(f"Epoch {epoch + 1}, {pair_type} Loss: {loss}")

        # Store and print overall loss
        overall_loss = pair_losses['overall']
        self.overall_losses.append(overall_loss)
        print(f"Epoch {epoch + 1}, Overall Loss: {overall_loss}")

    def on_train_end(self, logs=None):
        # At the end of training, save the loss plot
        # self.save_loss_plot()
        # self._save_plot()
        self.save_percent_plot()
        self.save_sep_sep_loss_vs_frequency()
        self.save_slope_of_loss_vs_frequency()
        self.save_combined_loss_plot()

    def find_sep_samples(self, y_train: ndarray) -> ndarray:
        """
        Identifies the indices of SEP samples in the training labels.

        :param y_train: The array of training labels.
        :return: The indices of SEP samples.
        """
        is_sep = y_train > self.sep_threshold
        return np.where(is_sep)[0]

    def find_sample_indices(self, y_train: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Identifies the indices of SEP, elevated, and background samples in the training labels.

        :param y_train: The array of training labels.
        :return: Three arrays containing the indices of SEP, elevated, and background samples respectively.
        """
        is_sep = y_train > self.sep_threshold
        is_elevated = (y_train > self.threshold) & (y_train <= self.sep_threshold)
        is_background = y_train <= self.threshold

        sep_indices = np.where(is_sep)[0]
        elevated_indices = np.where(is_elevated)[0]
        background_indices = np.where(is_background)[0]

        return sep_indices, elevated_indices, background_indices

    # def save_combined_loss_plot(self): """ Saves a combined plot of the losses for each pair type and the overall
    # loss. """ epochs = range(1, len(self.overall_losses) + 1) plt.figure() colors = ['blue', 'green', 'red',
    # 'cyan', 'magenta', 'yellow', 'black']  # Different colors for different curves pair_types = list(
    # self.pair_type_losses.keys()) + ['overall']
    #
    #     for i, pair_type in enumerate(pair_types):
    #         if pair_type == 'overall':
    #             losses = self.overall_losses
    #         else:
    #             losses = self.pair_type_losses[pair_type]
    #         plt.plot(epochs, losses, '-o', label=f'{pair_type} Loss', color=colors[i], markersize=3)
    #
    #     plt.title(f'Losses per Pair Type and Overall Loss Per Epoch, Batch Size {self.batch_size}')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.grid(True)
    #
    #     if self.save_tag:
    #         file_path = f"./investigation/combined_loss_plot_{self.save_tag}.png"
    #     else:
    #         file_path = "./investigation/combined_loss_plot.png"
    #     plt.savefig(file_path)
    #     plt.close()
    #     print(f"Saved combined loss plot at {file_path}")

    def save_combined_loss_plot(self):
        """
        Saves a combined plot of the losses for each pair type and the overall loss as separate subplots.
        """
        epochs = range(1, len(self.overall_losses) + 1)
        pair_types = list(self.pair_type_losses.keys()) + ['overall']
        num_subplots = len(pair_types)
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']  # Different colors for each subplot

        plt.figure(figsize=(15, 10))  # Adjust the figure size as needed
        for i, pair_type in enumerate(pair_types):
            plt.subplot(num_subplots, 1, i + 1)
            losses = self.pair_type_losses[pair_type] if pair_type != 'overall' else self.overall_losses
            plt.plot(epochs, losses, '-o', label=f'{pair_type} Loss', color=colors[i], markersize=3)
            plt.title(f'{pair_type} Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()  # Adjust subplots to fit into the figure area.

        file_name = f"combined_loss_plot_{self.save_tag}.png" if self.save_tag else "combined_loss_plot.png"
        file_path = f"./investigation/{file_name}"
        plt.savefig(file_path)
        plt.close()
        print(f"Saved combined loss plot at {file_path}")

    def save_percent_plot(self):
        # Plot the percentage of SEP-SEP pairs per epoch
        epochs = list(range(1, len(self.sep_sep_percentages) + 1))
        plt.figure()
        plt.plot(epochs, self.sep_sep_percentages, '-o', label='Percentage of SEP-SEP Pairs', markersize=3)
        plt.title(f'Percentage of SEP-SEP Pairs Per Epoch, Batch Size {self.batch_size}')
        plt.xlabel('Epoch')
        plt.ylabel('Percentage')
        plt.legend()
        plt.grid(True)
        # plt.show()  # or save the figure if preferred
        if self.save_tag:
            file_path = f"./investigation/percent_sep_sep_plot_{str(self.save_tag)}.png"
        else:
            file_path = f"./investigation/percent_sep_sep_plot.png"
        plt.savefig(file_path)
        plt.close()
        print(f"Saved plot at {file_path}")

    def save_sep_sep_loss_vs_frequency(self) -> None:
        """
        Plots the SEP-SEP loss against the SEP-SEP counts at the end of training.
        """
        plt.figure()
        plt.scatter(self.cumulative_sep_sep_counts, self.sep_sep_losses, c='blue', label='SEP-SEP Loss vs Frequency',
                    s=9)
        plt.title(f'SEP-SEP Loss vs Frequency, Batch Size {self.batch_size}')
        plt.xlabel('SEP-SEP Frequency')
        plt.ylabel('SEP-SEP Loss')
        plt.legend()
        plt.grid(True)

        if self.save_tag:
            file_path = f"./investigation/sep_sep_loss_vs_frequency_{self.save_tag}.png"
        else:
            file_path = "./investigation/sep_sep_loss_vs_frequency.png"

        plt.savefig(file_path)
        plt.close()
        print(f"Saved SEP-SEP Loss vs Counts plot at {file_path}")

    def save_slope_of_loss_vs_frequency(self) -> None:
        """
        Plots the slope of the change in SEP-SEP loss with respect to the change in SEP-SEP frequency vs epochs.
        """
        # Calculate the differences (delta) between consecutive losses and counts
        delta_losses = np.diff(self.sep_sep_losses)
        delta_counts = np.diff(self.cumulative_sep_sep_counts)

        # To avoid division by zero, we will replace zeros with a small value (epsilon)
        epsilon = 1e-8
        delta_counts = np.where(delta_counts == 0, epsilon, delta_counts)

        # Calculate the slope (change in loss / change in frequency)
        slopes = delta_losses / delta_counts

        # Prepare the epochs for x-axis, which are one less than the number of losses due to diff operation
        epochs = range(1, len(self.sep_sep_losses))

        plt.figure()
        plt.plot(epochs, slopes, '-o', label='Slope of SEP-SEP Loss vs Frequency', markersize=3)
        plt.title(f'Slope of SEP-SEP Loss vs Frequency Change Per Epoch, Batch Size {self.batch_size}')
        plt.xlabel('Epoch')
        plt.ylabel('Slope')
        plt.legend()
        plt.grid(True)

        if self.save_tag:
            file_path = f"./investigation/slope_sep_sep_loss_vs_frequency_{self.save_tag}.png"
        else:
            file_path = "./investigation/slope_sep_sep_loss_vs_frequency.png"

        plt.savefig(file_path)
        plt.close()
        print(f"Saved Slope of Loss vs Counts plot at {file_path}")

    # def _save_plot(self):
    #     plt.figure()
    #     plt.plot(self.epochs_10s, self.losses, '-o', label='Training Loss', markersize=3)
    #     plt.title(f'Training Loss, Batch Size {self.batch_size}')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.grid(True)
    #     if self.save_tag:
    #         file_path = f"./investigation/training_loss_plot_{str(self.save_tag)}.png"
    #     else:
    #         file_path = f"./investigation/training_loss_plot.png"
    #     plt.savefig(file_path)
    #     plt.close()
    #     print(f"Saved plot at {file_path}")

    # def save_loss_plot(self):
    #     """
    #     Saves a plot of the SEP loss at each epoch.
    #     """
    #     plt.figure()
    #     plt.plot(range(1, len(self.sep_sep_losses) + 1), self.sep_sep_losses, '-o', label='SEP Loss', markersize=3)
    #     plt.title(f'SEP Loss vs Batches, Batch Size {self.batch_size}')
    #     plt.xlabel('batches')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.grid(True)
    #     if self.save_tag:
    #         file_path = f"./investigation/sep_loss_plot_{self.save_tag}.png"
    #     else:
    #         file_path = "./investigation/sep_loss_plot.png"
    #     plt.savefig(file_path)
    #     plt.close()
    #     print(f"Saved SEP loss plot at {file_path}")


def pds_loss_eval(y_true, z_pred, reduction='none'):
    """
    Computes the loss for a batch of predicted features and their labels.

    :param y_true: A batch of true label values, shape of [batch_size, 1].
    :param z_pred: A batch of predicted Z values, shape of [batch_size, 2].
    :param reduction: The type of reduction to apply to the loss ('sum', 'none', or 'mean').
    :return: The average error for all unique combinations of the samples in the batch.
    """
    int_batch_size = len(z_pred)
    total_error = 0.0

    # print("received batch size in custom eval:", int_batch_size)

    # Loop through all unique pairs of samples in the batch
    for i in range(int_batch_size):
        for j in range(i + 1, int_batch_size):
            z1, z2 = z_pred[i], z_pred[j]
            label1, label2 = y_true[i], y_true[j]
            # Update pair counts (implement this function as needed)
            # update_pair_counts(label1, label2)
            err = error(z1, z2, label1, label2)  # Make sure 'error' function uses NumPy or standard Python
            total_error += err

    if reduction == 'sum':
        return total_error  # total loss
    elif reduction == 'none' or reduction == 'mean':
        denom = int_batch_size * (int_batch_size - 1) / 2 + 1e-9
        return total_error / denom  # average loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}.")


def pds_loss_eval_pairs(y_true, z_pred, reduction='none'):
    """
    Computes the loss for a batch of predicted features and their labels.
    Returns a dictionary of average losses for each pair type and overall.

    :param y_true: A batch of true label values, shape of [batch_size, 1].
    :param z_pred: A batch of predicted Z values, shape of [batch_size, 2].
    :param reduction: The type of reduction to apply to the loss ('sum', 'none', or 'mean').
    :return: A dictionary containing the average errors for all pair types and overall.
    """
    int_batch_size = len(z_pred)
    total_error = 0.0
    pair_errors = {
        'sep_sep': 0.0,
        'sep_elevated': 0.0,
        'sep_background': 0.0,
        'elevated_elevated': 0.0,
        'elevated_background': 0.0,
        'background_background': 0.0
    }
    pair_counts = {key: 0 for key in pair_errors.keys()}

    # print("Received batch size in custom eval:", int_batch_size)

    # Loop through all unique pairs of samples in the batch
    for i in range(int_batch_size):
        for j in range(i + 1, int_batch_size):
            z1, z2 = z_pred[i], z_pred[j]
            label1, label2 = y_true[i], y_true[j]

            # Determine the pair type
            pair_type = determine_pair_type(label1, label2)  # Implement this function
            err = error(z1, z2, label1, label2)  # Make sure 'error' function uses NumPy or standard Python
            pair_errors[pair_type] += err
            pair_counts[pair_type] += 1
            total_error += err

    # Apply reduction
    if reduction == 'sum':
        avg_pair_errors = {key: error_sum for key, error_sum in pair_errors.items()}
        avg_pair_errors['overall'] = total_error
    elif reduction == 'none' or reduction == 'mean':
        avg_pair_errors = {key: pair_errors[key] / pair_counts[key] if pair_counts[key] > 0 else 0 for key in
                           pair_errors}
        denom = int_batch_size * (int_batch_size - 1) / 2 + 1e-9
        avg_pair_errors['overall'] = total_error / denom
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}.")

    return avg_pair_errors


def determine_pair_type(label1, label2, sep_threshold=None, elevated_threshold=None):
    """
    Determines the pair type based on the labels.

    :param label1: The label of the first sample.
    :param label2: The label of the second sample.
    :param sep_threshold: The threshold to classify SEP samples.
    :param elevated_threshold: The threshold to classify elevated samples.
    :return: A string representing the pair type.
    """

    if sep_threshold is None:
        sep_threshold = np.log(10)

    if elevated_threshold is None:
        elevated_threshold = np.log(10.0 / np.exp(2))

    if label1 > sep_threshold and label2 > sep_threshold:
        return 'sep_sep'
    elif (label1 > sep_threshold and label2 > elevated_threshold) or (
            label2 > sep_threshold and label1 > elevated_threshold):
        return 'sep_elevated'
    elif (label1 > sep_threshold and label2 <= elevated_threshold) or (
            label2 > sep_threshold and label1 <= elevated_threshold):
        return 'sep_background'
    elif label1 > elevated_threshold and label2 > elevated_threshold:
        return 'elevated_elevated'
    elif (label1 > elevated_threshold >= label2) or (
            label2 > elevated_threshold >= label1):
        return 'elevated_background'
    else:
        return 'background_background'


def evaluate(model, X, y, batch_size=-1, pairs=False):
    """
    Custom evaluate function to compute loss over the dataset.

    :param model: The trained model.
    :param X: Input features.
    :param y: True labels.
    :param batch_size: Size of the batch, use the whole dataset if batch_size <= 0.
    :param pairs: If True, uses pds_loss_eval_pairs to evaluate loss on pairs.
    :return: Calculated loss over the dataset or a dictionary of losses for each pair type.
    """
    if batch_size <= 0:
        z_pred = model.predict(X)
        return pds_loss_eval_pairs(y, z_pred) if pairs else pds_loss_eval(y, z_pred)

    total_loss = 0
    pair_losses = {key: 0.0 for key in
                   ['sep_sep', 'sep_elevated', 'sep_background', 'elevated_elevated', 'elevated_background',
                    'background_background']}
    pair_counts = {key: 0 for key in pair_losses}
    total_batches = 0

    for i in range(0, len(X), batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        z_pred = model.predict(X_batch)

        if pairs:
            batch_pair_losses = pds_loss_eval_pairs(y_batch, z_pred)
            for key in batch_pair_losses:
                pair_losses[key] += batch_pair_losses[key]
                pair_counts[key] += 1  # Count each batch for each pair type
        else:
            total_loss += pds_loss_eval(y_batch, z_pred)

        total_batches += 1

    if pairs:
        # Compute average losses for each pair type
        avg_pair_losses = {key: pair_losses[key] / pair_counts[key] if pair_counts[key] > 0 else 0 for key in
                           pair_losses}
        return avg_pair_losses

    return total_loss / total_batches if total_batches > 0 else 0


# def evaluate(model, X, y, batch_size=-1):
#     """
#     Custom evaluate function to compute loss over the dataset.
#
#     :param model: The trained model.
#     :param X: Input features.
#     :param y: True labels.
#     :param batch_size: Size of the batch, use the whole dataset if batch_size <= 0.
#     :return: Calculated loss over the dataset.
#     """
#     total_loss = 0
#     total_batches = 0
#
#     # print batch size received
#     print(f'batch size received: {batch_size}')
#
#     if batch_size <= 0:
#         # Use the whole dataset
#         z_pred = model.predict(X)
#         total_loss = pds_loss_eval(y, z_pred, reduction='none')
#         total_batches = 1
#     else:
#         # Process in batches
#         for i in range(0, len(X), batch_size):
#             X_batch = X[i:i + batch_size]
#             y_batch = y[i:i + batch_size]
#             z_pred = model.predict(X_batch)  # model prediction
#             batch_loss = pds_loss_eval(y_batch, z_pred, reduction='none')
#             total_loss += batch_loss
#             total_batches += 1
#
#     average_loss = total_loss / total_batches
#     return average_loss


# Helper function to map 2D indices to 1D indices (assuming it's defined elsewhere in your code)
# def map_to_1D_idx(i, j, n):
#     return n * i + j


# main run
if __name__ == '__main__':

    print("Testing the vectorized loss function...")
    print("WITHOUT SAMPLE WEIGHTS")
    loss_tester = ModelBuilder()
    # Generate dummy data for testing
    # np.random.seed(42)  # For reproducibility
    # batch_size = 200
    # z_dim = 9
    # y_true_dummy = np.random.rand(batch_size, 1).astype(np.float32)
    # z_pred_dummy = np.random.rand(batch_size, z_dim).astype(np.float32)
    #
    # print("y_true_dummy shape:", y_true_dummy.shape)
    # print("z_pred_dummy shape:", z_pred_dummy.shape)
    #
    # # Convert NumPy arrays to TensorFlow tensors
    # y_true_tensor = tf.convert_to_tensor(y_true_dummy, dtype=tf.float32)
    # z_pred_tensor = tf.convert_to_tensor(z_pred_dummy, dtype=tf.float32)
    #
    # # Time and compute loss using the original function
    # print("Computing loss using the original function...")
    # start_time_original = time.time()
    # loss_original = loss_tester.pds_loss(y_true_tensor, z_pred_tensor)
    # end_time_original = time.time()
    # original_duration = end_time_original - start_time_original
    #
    # # Time and compute loss using the vectorized function
    # print("Computing loss using the vectorized function...")
    # start_time_vectorized = time.time()
    # loss_vectorized = loss_tester.pds_loss_vec(y_true_tensor, z_pred_tensor)
    # end_time_vectorized = time.time()
    # vectorized_duration = end_time_vectorized - start_time_vectorized
    #
    # # Evaluate the TensorFlow tensors to get their numpy values
    # loss_original_value = loss_original.numpy()
    # loss_vectorized_value = loss_vectorized.numpy()
    #
    # # Print the losses and timing for comparison
    # print(f"Original Loss: {loss_original_value}, Time Taken: {original_duration} seconds")
    # print(f"Vectorized Loss: {loss_vectorized_value}, Time Taken: {vectorized_duration} seconds")
    #
    # # # Check if the losses are approximately equal
    # # np.testing.assert_almost_equal(loss_original_value, loss_vectorized_value, decimal=5)
    # # print("Test passed: The original and vectorized loss functions return approximately the same value.")
    #
    # # Compare the execution time
    # if vectorized_duration < original_duration:
    #     print(f"The vectorized function is faster by {original_duration - vectorized_duration} seconds.")
    # else:
    #     print(f"The original function is faster by {vectorized_duration - original_duration} seconds.")

    print("WITH SAMPLE WEIGHTS")
    # Generate dummy data for testing
    np.random.seed(42)  # For reproducibility
    batch_size = 200
    z_dim = 9
    num_unique_pairs = batch_size * (batch_size - 1) // 2
    y_true_dummy = np.random.rand(batch_size, 1).astype(np.float32)
    z_pred_dummy = np.random.rand(batch_size, z_dim).astype(np.float32)
    sample_weights_dummy = np.random.rand(num_unique_pairs, 1).astype(np.float32)

    print("y_true_dummy shape:", y_true_dummy.shape)
    print("z_pred_dummy shape:", z_pred_dummy.shape)
    print("sample_weights_dummy shape:", sample_weights_dummy.shape)

    # Convert NumPy arrays to TensorFlow tensors
    y_true_tensor = tf.convert_to_tensor(y_true_dummy, dtype=tf.float32)
    z_pred_tensor = tf.convert_to_tensor(z_pred_dummy, dtype=tf.float32)
    sample_weights_tensor = tf.convert_to_tensor(sample_weights_dummy, dtype=tf.float32)  # Convert sample weights

    # Time and compute loss using the original function with sample weights
    print("Computing loss using the original function with sample weights...")
    start_time_original = time.time()
    loss_original = loss_tester.pds_loss_dl(y_true_tensor, z_pred_tensor, sample_weights=sample_weights_tensor)
    end_time_original = time.time()
    original_duration = end_time_original - start_time_original

    # Time and compute loss using the vectorized function with sample weights
    print("Computing loss using the vectorized function with sample weights...")
    start_time_vectorized = time.time()
    loss_vectorized = loss_tester.pds_loss_dl_vec(y_true_tensor, z_pred_tensor, sample_weights=sample_weights_tensor)
    end_time_vectorized = time.time()
    vectorized_duration = end_time_vectorized - start_time_vectorized

    # Evaluate the TensorFlow tensors to get their numpy values
    loss_original_value = loss_original.numpy()
    loss_vectorized_value = loss_vectorized.numpy()

    # Print the losses and timing for comparison
    print(f"Original Loss with Sample Weights: {loss_original_value}, Time Taken: {original_duration} seconds")
    print(f"Vectorized Loss with Sample Weights: {loss_vectorized_value}, Time Taken: {vectorized_duration} seconds")

    # Compare the execution time
    if vectorized_duration < original_duration:
        print(
            f"The vectorized function with sample weights is faster by {original_duration - vectorized_duration} seconds.")
    else:
        print(
            f"The original function with sample weights is faster by {vectorized_duration - original_duration} seconds.")
