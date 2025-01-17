import os
import random
import traceback
from collections import Counter
from pathlib import Path
from typing import Generator, Tuple, Optional
from typing import List, Union, Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib.lines import Line2D
from numpy import ndarray
from scipy import stats
from scipy.signal import correlate, correlation_lags
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Flatten,
    Dense,
    Concatenate,
    GRU,
    Dropout,
    LeakyReLU,
    BatchNormalization,
    LayerNormalization,
    MaxPooling1D,
    AveragePooling1D,
    Add,
    Multiply,
    Lambda,
    Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.regularizers import l2

from modules.shared.globals import PLUS_INDEX, MID_INDEX, MINUS_INDEX, MLP_HIDDENS, UPPER_THRESHOLD_MOE, \
    LOWER_THRESHOLD_MOE
from modules.training.normlayer import NormalizeLayer
from modules.training.phase_manager import TrainingPhaseManager, create_weight_tensor_fast
from modules.training.sam_keras import SAMModel

# Seeds for reproducibility
seed_value = 42

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)


def get_plus_cls(
        X: np.ndarray,
        y: np.ndarray,
        upper_threshold: float = UPPER_THRESHOLD_MOE,
        *additional_sets
) -> tuple[np.ndarray, ...]:
    """
    Get data samples where y values are >= upper threshold (plus class).

    Args:
        X (np.ndarray): Input features array of shape (n_samples, n_features)
        y (np.ndarray): Target values array of shape (n_samples, n_outputs) 
        upper_threshold (float): Upper threshold value to filter samples
        *additional_sets: Additional arrays to filter using the same mask

    Returns:
        tuple[np.ndarray, ...]: Filtered X, y and any additional arrays containing only plus class samples
    """
    plus_mask = y[:, 0] >= upper_threshold
    filtered = [X[plus_mask], y[plus_mask]]
    filtered.extend(arr[plus_mask] for arr in additional_sets)
    return tuple(filtered)


def get_zero_cls(
        X: np.ndarray,
        y: np.ndarray,
        lower_threshold: float = LOWER_THRESHOLD_MOE,
        upper_threshold: float = UPPER_THRESHOLD_MOE,
        *additional_sets) -> tuple[np.ndarray, ...]:
    """
    Get data samples where y values are between the lower and upper thresholds (zero class).

    Args:
        X (np.ndarray): Input features array of shape (n_samples, n_features)
        y (np.ndarray): Target values array of shape (n_samples, n_outputs)
        lower_threshold (float): Lower threshold value to filter samples
        upper_threshold (float): Upper threshold value to filter samples
        *additional_sets: Additional arrays to filter using the same mask

    Returns:
        tuple[np.ndarray, ...]: Filtered X, y and any additional arrays containing only zero class samples
    """
    zero_mask = (y[:, 0] > lower_threshold) & (y[:, 0] < upper_threshold)
    filtered = [X[zero_mask], y[zero_mask]]
    filtered.extend(arr[zero_mask] for arr in additional_sets)
    return tuple(filtered)


def get_minus_cls(
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = LOWER_THRESHOLD_MOE,
        *additional_sets
) -> tuple[np.ndarray, ...]:
    """
    Get data samples where y values are below or equal to the threshold (minus class).

    Args:
        X (np.ndarray): Input features array of shape (n_samples, n_features)
        y (np.ndarray): Target values array of shape (n_samples, n_outputs)
        threshold (float): Threshold value to filter samples
        *additional_sets: Additional arrays to filter using the same mask

    Returns:
        tuple[np.ndarray, ...]: Filtered X, y and any additional arrays containing only minus class samples
    """
    minus_mask = y[:, 0] <= threshold
    filtered = [X[minus_mask], y[minus_mask]]
    filtered.extend(arr[minus_mask] for arr in additional_sets)
    return tuple(filtered)


def convert_to_onehot_cls(
        y: np.ndarray,
        lower_threshold: float = LOWER_THRESHOLD_MOE,
        upper_threshold: float = UPPER_THRESHOLD_MOE
) -> np.ndarray:
    """
    Convert regression values to one-hot encoded classes based on thresholds.

    Args:
        y (np.ndarray): Input array of shape (n_samples, n_features) containing regression values
        lower_threshold (float): Lower threshold for class separation
        upper_threshold (float): Upper threshold for class separation

    Returns:
        np.ndarray: One-hot encoded classes of shape (n_samples, 3) where:
            - index 0 = high class (values >= upper_threshold)
            - index 1 = mid class (lower_threshold < values < upper_threshold) 
            - index 2 = low class (values <= lower_threshold)
    """
    y_classes = np.zeros((y.shape[0], 3))
    y_classes[y[:, 0] >= upper_threshold, 0] = 1  # High class
    y_classes[(y[:, 0] > lower_threshold) & (y[:, 0] < upper_threshold), 1] = 1  # Mid class
    y_classes[y[:, 0] <= lower_threshold, 2] = 1  # Low class
    return y_classes


def create_1dcnn(
        input_dims: list,
        hiddens: List[tuple],
        embed_dim: int = 50,
        output_dim: int = 1,
        pds: bool = False,
        l2_reg: float = None,
        dropout: float = 0.0,
        activation=None,
        norm: str = None,
        name: str = '1dcnn'
) -> Model:
    """
    Create a CNN model with multiple input branches, each processing inputs stacked across the channel dimension.

    Parameters:
    - input_dims (list): List of input dimensions. Groups of similar dimensions represent separate channels.
    - hiddens (list[tuple]): List of tuples for CNN layers configuration.
      Each tuple contains (filters, kernel_size, dilation_rate, pooling_type).
    - embed_dim (int): The number of units in the fully connected layer. Default is 50.
    - output_dim (int): The dimension of the output layer. Default is 1 for regression tasks.
    - pds (bool): If True, the model will be use PDS and there will have its representations normalized.

    Returns:
    - Model: A Keras model instance.
    """

    if input_dims is None:
        input_dims = [25, 25, 25]

    dim_counts = Counter(input_dims)
    branches = []
    cnn_inputs = []

    for dim, count in dim_counts.items():
        input_layer = Input(shape=(dim, count), name=f'input_{dim}x{count}')
        x = input_layer

        for filters, kernel_size, dilation_rate, pool_type, pool_size in hiddens:
            x = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                dilation_rate=dilation_rate,
                kernel_regularizer=l2(l2_reg) if l2_reg else None)(x)

            if norm == 'batch_norm':
                x = BatchNormalization()(x)
            elif norm == 'layer_norm':
                x = LayerNormalization()(x)

            x = activation(x) if callable(activation) else LeakyReLU()(x)

            if dropout > 0.0:
                x = Dropout(dropout)(x)

            if pool_type == 'max':
                x = MaxPooling1D(pool_size=pool_size)(x)
            elif pool_type == 'avg':
                x = AveragePooling1D(pool_size=pool_size)(x)

        flattened = Flatten()(x)
        branches.append(flattened)
        cnn_inputs.append(input_layer)

    concatenated = Concatenate()(branches) if len(branches) > 1 else branches[0]

    dense = Dense(embed_dim, kernel_regularizer=l2(l2_reg) if l2_reg else None)(concatenated)
    if pds:
        normalized_repr_layer = NormalizeLayer(name='normalize_layer')(dense)
        final_repr_output = normalized_repr_layer
    else:
        final_repr_output = activation(dense) if callable(activation) else LeakyReLU()(dense)

    if output_dim > 0:
        output_layer = Dense(output_dim, name='forecast_head')(final_repr_output)
        model_output = [final_repr_output, output_layer]
    else:
        model_output = final_repr_output

    model = Model(inputs=cnn_inputs, outputs=model_output, name=name)
    return model


def create_gru_with_addition_skips(
        input_dims: list = None,
        gru_units: int = 64,
        gru_layers: int = 2,
        embed_dim: int = 50,
        output_dim: int = 1,
        pds: bool = False,
        l2_reg: float = None,
        dropout: float = 0.0,
        activation=None,
        norm: str = None,
        skipped_layers: int = 2,  # New parameter for skip connections
        name: str = 'gru'
) -> Model:
    """
    Create a GRU model with addition-based skip connections.
    """

    if input_dims is None:
        input_dims = [25, 25, 25]  # Default input dimensions

    if activation is None:
        activation = LeakyReLU()  # Default to LeakyReLU if no activation is specified

    dim_counts = Counter(input_dims)
    rnn_branches = []
    gru_inputs = []

    for dim, count in dim_counts.items():
        input_layer = Input(shape=(None, count), name=f'input_{dim}x{count}')
        x = input_layer
        skip_connection = None  # Initialize skip connection storage

        for layer in range(gru_layers):
            if layer % skipped_layers == 0 and layer > 0:
                # Prepare for addition-based skip connection
                if skip_connection is not None:
                    x = Add()([x, skip_connection])

            x = GRU(units=gru_units,
                    return_sequences=True if layer < gru_layers - 1 else False,
                    kernel_regularizer=l2(l2_reg) if l2_reg else None)(x)

            if norm == 'batch_norm':
                x = BatchNormalization()(x)
            elif norm == 'layer_norm':
                x = LayerNormalization()(x)

            if callable(activation):
                x = activation(x)
            else:
                x = LeakyReLU()(x)  # Fallback if activation is not callable, for safety

            if dropout > 0.0:
                x = Dropout(dropout)(x)

            if layer % skipped_layers == 0 or skip_connection is None:
                skip_connection = x  # Update skip connection after applying activation

        flattened = Flatten()(x)
        rnn_branches.append(flattened)
        gru_inputs.append(input_layer)

    concatenated = Concatenate()(rnn_branches) if len(rnn_branches) > 1 else rnn_branches[0]

    dense = Dense(embed_dim, kernel_regularizer=l2(l2_reg) if l2_reg else None)(concatenated)
    if callable(activation):
        final_repr_output = activation(dense)
    else:
        final_repr_output = LeakyReLU()(dense)

    if pds:
        # Assuming NormalizeLayer is defined elsewhere
        normalized_repr_layer = NormalizeLayer(name='normalize_layer')(
            final_repr_output)  # Custom normalization layer for PDS
        final_repr_output = normalized_repr_layer

    if output_dim > 0:
        output_layer = Dense(output_dim, name='forecast_head')(final_repr_output)
        model_output = [final_repr_output, output_layer]
    else:
        model_output = final_repr_output

    return Model(inputs=gru_inputs, outputs=model_output, name=name)


def create_gru_with_concatenation_skips(
        input_dims: list = None,
        gru_units: int = 64,
        gru_layers: int = 2,
        embed_dim: int = 50,
        output_dim: int = 1,
        pds: bool = False,
        l2_reg: float = None,
        dropout: float = 0.0,
        activation=None,
        norm: str = None,
        skipped_layers: int = 2,  # New parameter for skip connections
        name: str = 'gru_concat'
) -> Model:
    """
    Create a GRU model with concatenation-based skip connections.
    """
    if input_dims is None:
        input_dims = [25, 25, 25]  # Default input dimensions

    if activation is None:
        activation = LeakyReLU  # Default to LeakyReLU if no activation is specified

    dim_counts = Counter(input_dims)
    rnn_branches = []
    gru_inputs = []

    for dim, count in dim_counts.items():
        input_layer = Input(shape=(None, count), name=f'input_{dim}x{count}')
        x = input_layer
        skip_connection = None  # Initialize skip connection storage

        for layer in range(gru_layers):
            if layer % skipped_layers == 0 and layer > 0 and skip_connection is not None:
                # Concatenate skip connection with current layer output
                x = Concatenate()([x, skip_connection])
                # After concatenation, you might need to adjust the dimensionality
                # For simplicity, this is not included here but could involve a Dense layer or adjusting GRU units

            x = GRU(units=gru_units,
                    return_sequences=True if layer < gru_layers - 1 else False,
                    kernel_regularizer=l2(l2_reg) if l2_reg else None)(x)

            if norm == 'batch_norm':
                x = BatchNormalization()(x)
            elif norm == 'layer_norm':
                x = LayerNormalization()(x)

            if callable(activation):
                x = activation()(x)
            else:
                x = LeakyReLU()(x)  # Fallback if activation is not callable, for safety

            if dropout > 0.0:
                x = Dropout(dropout)(x)

            if layer % skipped_layers == 0:
                skip_connection = x  # Update skip connection to current x

        flattened = Flatten()(x)
        rnn_branches.append(flattened)
        gru_inputs.append(input_layer)

    concatenated = Concatenate()(rnn_branches) if len(rnn_branches) > 1 else rnn_branches[0]

    # Adjust the Dense layer's input size if necessary due to concatenation
    dense = Dense(embed_dim, kernel_regularizer=l2(l2_reg) if l2_reg else None)(concatenated)
    if callable(activation):
        final_repr_output = activation()(dense)
    else:
        final_repr_output = LeakyReLU()(dense)

    if pds:
        # Assuming NormalizeLayer is a custom layer defined elsewhere
        normalized_repr_layer = NormalizeLayer(name='normalize_layer')(final_repr_output)
        final_repr_output = normalized_repr_layer

    if output_dim > 0:
        output_layer = Dense(output_dim, name='forecast_head')(final_repr_output)
        model_output = [final_repr_output, output_layer]
    else:
        model_output = final_repr_output

    return Model(inputs=gru_inputs, outputs=model_output, name=name)


def create_gru(
        input_dim: int = 100,  # Number of features
        gru_units: int = 30,
        gru_layers: int = 1,
        embed_dim: int = 128,
        output_dim: int = 1,
        pds: bool = False,
        dropout: float = 0.0,
        activation=None,
        norm: str = None,
        name: str = 'gru'
) -> Model:
    """
    Create a GRU model that processes time series data.

    Args:
        input_dim (int): Dimension of the input features. Default is 100.
        gru_units (int): Number of units in each GRU layer. Default is 30.
        gru_layers (int): Number of GRU layers to stack. Default is 1.
        embed_dim (int): Dimension of the representation layer. Default is 128.
        output_dim (int): Dimension of the output layer. If 0, only returns representation. Default is 1.
        pds (bool): Whether to apply normalization to representation layer. Default is False.
        dropout (float): Dropout rate to apply after each GRU layer. Default is 0.0.
        activation: Activation function to use. If None, uses LeakyReLU. Default is None.
        norm (str): Type of normalization to use ('batch_norm' or 'layer_norm'). Default is None.
        name (str): Name of the model. Default is 'gru'.

    Returns:
        Model: A Keras Model instance with the specified GRU architecture.

    The model architecture consists of:
    1. An input layer accepting a sequence of shape (input_dim,)
    2. Reshaping to (timesteps, features) for GRU processing
    3. One or more GRU layers with optional normalization and dropout
    4. A dense representation layer
    5. Optional normalization of the representation
    6. Optional output layer
    """
    # Input shape should be (input_dim,)
    input_layer = Input(shape=(input_dim,))

    # Reshape to (timesteps, features)
    timesteps = 25
    features = input_dim // timesteps  # Should be 4 when input_dim is 100
    x = Reshape((timesteps, features), name='reshape_layer')(input_layer)

    for layer in range(gru_layers):
        x = GRU(units=gru_units,
                return_sequences=True if layer < gru_layers - 1 else False)(x)

        if norm == 'batch_norm':
            x = BatchNormalization()(x)
        elif norm == 'layer_norm':
            x = LayerNormalization()(x)

        x = activation(x) if callable(activation) else LeakyReLU()(x)

        if dropout > 0.0:
            x = Dropout(dropout)(x)

    # Dense layer for representation
    dense = Dense(embed_dim)(x)
    final_repr_output = activation(dense) if callable(activation) else LeakyReLU()(dense)

    if pds:
        normalized_repr_layer = NormalizeLayer(name='normalize_layer')(final_repr_output)
        final_repr_output = normalized_repr_layer

    if output_dim > 0:
        output_layer = Dense(output_dim, name='forecast_head')(final_repr_output)
        model_output = [final_repr_output, output_layer]
    else:
        model_output = final_repr_output

    model = Model(inputs=input_layer, outputs=model_output, name=name)
    return model


# TODO: generalize this function to handle 
def load_partial_weights_from_path(
        pretrained_weights_path: str,
        new_model: Model,
        old_model_params: Dict,
        pretraining: bool = False,
        skip_layers: Optional[List[str]] = None
) -> None:
    """
    Load weights from a pre-trained model (via a weights file path) into an existing new model,
    transferring compatible layers and skipping any layers that differ (such as the final output layer).

    This utility:
    - Creates the old model using the provided parameters in `old_model_params`.
    - Loads the old model's weights from `pretrained_weights_path`.
    - Copies matching layer weights into `new_model`.
    - Skips loading weights for layers specified in `skip_layers` or any that are incompatible by shape.

    Parameters
    ----------
    pretrained_weights_path : str
        Path to the file containing the pre-trained model's weights. Typically something like 'model_weights.h5'.

    new_model : Model
        The new Keras model instance into which compatible weights will be loaded.
        This model may have a different output dimension than the original model.

    old_model_params : Dict
        A dictionary containing the parameters needed to recreate the old model.
        For example:
        {
            'input_dim': <int>,
            'hiddens': <list of ints>,
            'output_dim': <int>,    # The original output dimension
            'pretraining': <bool>,
            'embed_dim': <int>,
            'dropout': <float>,
            'activation': <callable or None>,
            'norm': <str>,
            'skip_repr': <bool>,
            'skipped_layers': <int>,
            'sam_rho': <float>
        }
        Make sure these match exactly the parameters used when the pre-trained model was saved.

    pretraining : bool
        If True, the model is pretraining and the weights are loaded from a PDC/PDS pre-trained model.
        If False, the model is not pretraining and the weights are loaded from a trained model.
        
    skip_layers : Optional[List[str]]
        A list of layer names to skip while loading weights. For example, ['forecast_head']
        if the output layer's shape has changed.

    Returns
    -------
    None
        The function modifies `new_model` in-place by setting the weights of its matching layers
        to those of the old model where compatible.
    """
    if skip_layers is None:
        skip_layers = []

    # Import the same create_mlp function used to create the old model
    # Ensure that this matches the original environment:
    # from your_module import create_mlp

    # Create the old model with identical parameters used originally
    old_model = create_mlp(
        input_dim=old_model_params['input_dim'],
        hiddens=old_model_params['hiddens'],
        output_dim=old_model_params['output_dim'],  # original output dimension
        pretraining=old_model_params['pretraining'],
        embed_dim=old_model_params['embed_dim'],
        dropout=old_model_params['dropout'],
        activation=old_model_params['activation'],
        norm=old_model_params['norm'],
        skip_repr=old_model_params['skip_repr'],
        skipped_layers=old_model_params['skipped_layers'],
        sam_rho=old_model_params['sam_rho']
    )

    if pretraining:
        old_model = add_proj_head(old_model)

    # Load the old model's weights
    old_model.load_weights(pretrained_weights_path)

    # Map old model's layers by name for easy access
    old_layers_dict = {layer.name: layer for layer in old_model.layers}

    # Transfer weights from old_model to new_model
    for new_layer in new_model.layers:
        if new_layer.name in skip_layers:
            continue

        if new_layer.name in old_layers_dict:
            old_layer = old_layers_dict[new_layer.name]
            old_weights = old_layer.get_weights()
            new_weights = new_layer.get_weights()

            # Ensure the layer has weights and they match in count
            if len(old_weights) == len(new_weights):
                # Verify that corresponding weight arrays have the same shape
                if all(o_w.shape == n_w.shape for o_w, n_w in zip(old_weights, new_weights)):
                    new_layer.set_weights(old_weights)
                # else shapes differ, skip loading this layer
            # else number of weights differ, skip this layer
        # else no corresponding old layer by that name, skip

    # After this, `new_model` will have partially loaded weights, except for layers that didn't match or were skipped.


def create_mlp(
        input_dim: int = 100,
        output_dim: int = 1,
        hiddens=None,
        skipped_layers: int = 1,
        embed_dim: int = 128,
        skip_repr: bool = True,
        pretraining: bool = False,
        activation=None,
        norm: str = 'batch_norm',
        sam_rho: float = 1e-2,
        dropout: float = 0.2,
        output_activation=None,
        name: str = 'mlp'
) -> Model:
    """
    Create an MLP model with fully connected dense layers and configurable activation functions.
    Residual connections are automatically added when skipped_layers > 0.

    Parameters:
    - input_dim (int): The number of features in the input data.
    - output_dim (int): The dimension of the output layer. Default is 1 for regression tasks.
    - hiddens (list): A list of integers where each integer is the number of units in a hidden layer.
    - skipped_layers (int): Number of layers between residual connections. If 0, no residual connections.
                           Must be less than the total number of hidden layers.
    - embed_dim (int): The number of features in the final representation vector.
    - skip_repr (bool): If True and skipped_layers > 0, adds a residual connection to the representation layer.
    - pretraining (bool): If True, the model will use PDS/PDC and there will have its representations normalized.
    - activation: Optional activation function to use. If None, defaults to LeakyReLU.
    - norm (str): Optional normalization type to use ('batch_norm' or 'layer_norm'). Default is None.
    - sam_rho (float): Size of the neighborhood for perturbation in SAM. Default is 0.05. If 0.0, SAM is not used.
    - dropout (float): Dropout rate to apply after activations or residual connections. If 0.0, no dropout is applied.
    - output_activation: Optional activation function for output layer. Use 'softmax' for multi-class or 'sigmoid' for binary.
    
    Returns:
    - Model: A Keras model instance.

    Raises:
    - ValueError: If skipped_layers is greater than or equal to the number of hidden layers.
    """
    # Set default hidden layers if none provided
    if hiddens is None:
        hiddens = [50, 50]

    # Validate skipped_layers parameter
    if skipped_layers >= len(hiddens):
        raise ValueError(
            f"skipped_layers ({skipped_layers}) must be less than the number of hidden layers ({len(hiddens)})"
        )

    # Create input layer
    input_layer = Input(shape=(input_dim,))
    has_residuals = skipped_layers > 0

    # First block (special case to ensure proper skip from input)
    x = Dense(hiddens[0])(input_layer)
    if norm == 'batch_norm':
        x = BatchNormalization()(x)
    x = activation(x) if callable(activation) else LeakyReLU()(x)

    # First skip connection (from input)
    if has_residuals:
        if x.shape[-1] != input_layer.shape[-1]:
            residual_proj = Dense(x.shape[-1], use_bias=False)(input_layer)
        else:
            residual_proj = input_layer
        x = Add()([x, residual_proj])
        if norm == 'layer_norm':
            x = LayerNormalization()(x)
        # Add dropout after residual connection if dropout > 0
        if dropout > 0:
            x = Dropout(dropout)(x)
    elif dropout > 0:  # No residuals, add dropout after activation
        x = Dropout(dropout)(x)

    residual_layer = x

    # Remaining hidden layers
    for i, units in enumerate(hiddens[1:], start=1):
        # Dense + Norm + Activation block
        x = Dense(units)(x)
        if norm == 'batch_norm':
            x = BatchNormalization()(x)
        x = activation(x) if callable(activation) else LeakyReLU()(x)

        # Add skip connection if at a skip point
        if has_residuals and i % skipped_layers == 0:
            # Project the residual layer if needed
            if x.shape[-1] != residual_layer.shape[-1]:
                residual_proj = Dense(x.shape[-1], use_bias=False)(residual_layer)
            else:
                residual_proj = residual_layer
            x = Add()([x, residual_proj])
            if norm == 'layer_norm':
                x = LayerNormalization()(x)
            # Add dropout after residual connection if dropout > 0
            if dropout > 0:
                x = Dropout(dropout)(x)
            residual_layer = x
        elif dropout > 0:  # No residuals, add dropout after activation
            x = Dropout(dropout)(x)

    # Create final representation layer
    x = Dense(embed_dim)(x)
    if norm == 'batch_norm':
        x = BatchNormalization()(x)

    # Apply activation with appropriate naming based on skip_repr
    if skip_repr:  # if skip_repr, then the add down the line is the repr layer
        x = activation(x) if callable(activation) else LeakyReLU()(x)
    else:  # if no skip_repr, then the activation is repr layer
        x = activation(x) if callable(activation) else LeakyReLU(name='repr_layer')(x)

    # Add final skip connection if enabled
    if skip_repr and has_residuals:
        if x.shape[-1] != residual_layer.shape[-1]:
            residual_proj = Dense(embed_dim, use_bias=False)(residual_layer)
        else:
            residual_proj = residual_layer
        x = Add(name='repr_layer')([x, residual_proj])
        if norm == 'layer_norm':
            x = LayerNormalization()(x)
        # Add dropout after final residual connection if dropout > 0
        if dropout > 0:
            x = Dropout(dropout)(x)
    elif dropout > 0:  # No residuals, add dropout after activation
        x = Dropout(dropout)(x)
    elif norm == 'layer_norm':
        x = LayerNormalization()(x)
        x = Dropout(dropout)(x)

    # Handle PDS normalization if needed
    if pretraining:
        final_repr_output = NormalizeLayer(name='normalize_layer')(x)
    else:
        final_repr_output = x

    # Add output layer if output_dim > 0
    if output_dim > 0:
        output_layer = Dense(output_dim, activation=output_activation, name='forecast_head')(final_repr_output)
        model_output = [final_repr_output, output_layer]
    else:
        model_output = final_repr_output

    # Create appropriate model type based on SAM parameter
    if sam_rho > 0.0:
        model = SAMModel(inputs=input_layer, outputs=model_output, rho=sam_rho, name=name)
    else:
        model = Model(inputs=input_layer, outputs=model_output, name=name)

    return model


def add_proj_head(
        model: Model,
        output_dim: int = 1,
        hiddens: Optional[List[int]] = None,
        freeze_features: bool = True,
        pretraining: bool = False,
        dropout: float = 0.0,
        activation=None,
        norm: str = None,
        skipped_layers: int = 2,
        name: str = 'mlp',
        sam_rho: float = 0.05,
        output_activation: str = 'linear'
) -> Model:
    """
    Add a projection head with output layer to an existing model,
    replacing the existing prediction layer and optionally the decoder layer.

    :param model: The existing model
    :param output_dim: The dimensionality of the output layer.
    :param freeze_features: Whether to freeze the layers of the base model or not.
    :param hiddens: List of integers representing the hidden layers for the projection.
    :param pretraining: Whether to adapt the model for pretraining representations.
    :param dropout: Dropout rate for adding dropout layers.
    :param activation: Activation function to use. If None, defaults to LeakyReLU.
    :param norm: Type of normalization ('batch_norm' or 'layer_norm').
    :param skipped_layers: Number of layers between residual connections.
    :param name: Name of the model.
    :param sam_rho: Rho value for sharpness-aware minimization (SAM). Default is 0.05. if 0.0, SAM is not used.
    :param output_activation: Activation function for output layer. Default 'linear' for regression, use 'softmax' for classification.
    :return: The modified model with a projection layer and output layer.
    """

    if hiddens is None:
        hiddens = [6]

    if activation is None:
        activation = LeakyReLU()

    residual = True if skipped_layers > 0 else False

    print(f'Features are frozen: {freeze_features}')

    # Determine the layer to be kept based on whether pretraining representations are used
    layer_to_keep = 'normalize_layer' if pretraining else 'repr_layer'

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
    residual_layer = None

    for i, nodes in enumerate(hiddens):
        if i % skipped_layers == 0 and i > 0 and residual:
            if residual_layer is not None:
                # Check if projection is needed
                if x_proj.shape[-1] != residual_layer.shape[-1]:
                    # Correct projection to match 'x_proj' dimensions
                    residual_layer = Dense(x_proj.shape[-1], use_bias=False)(residual_layer)
                x_proj = Add()([x_proj, residual_layer])
            residual_layer = x_proj  # Update the starting point for the next residual connection
        else:
            if i % skipped_layers == 0 or residual_layer is None:
                residual_layer = x_proj

        x_proj = Dense(
            nodes,
            name=f"projection_layer_{i + 1}")(x_proj)

        if norm == 'batch_norm':
            x_proj = BatchNormalization(name=f"batch_norm_{i + 1}")(x_proj)
        elif norm == 'layer_norm':
            x_proj = LayerNormalization(name=f"layer_norm_{i + 1}")(x_proj)

        if callable(activation):
            x_proj = activation(x_proj)
        else:
            x_proj = LeakyReLU(name=f"activation_{i + 1}")(x_proj)

        if dropout > 0.0:
            x_proj = Dropout(dropout, name=f"proj_dropout_{dropout_count + i + 1}")(x_proj)

    # Add output layer with specified activation
    output_layer = Dense(output_dim, activation=output_activation, name=f"forecast_head")(x_proj)

    if sam_rho > 0.0:
        # create the new extended SAM model
        extended_model = SAMModel(inputs=new_base_model.input, outputs=[repr_output, output_layer], rho=sam_rho,
                                  name=name)
    else:
        # Create the new extended model
        extended_model = Model(inputs=new_base_model.input, outputs=[repr_output, output_layer], name=name)

    # If freeze_features is False, make all layers trainable
    if not freeze_features:
        for layer in extended_model.layers:
            layer.trainable = True

    return extended_model


def add_decoder(encoder_model, hiddens, activation=None, norm=None, dropout=0.0, skip_connections=False):
    """
    Adds a decoder to the given encoder model, reversing the encoder architecture.

    Parameters:
    - encoder_model: The encoder model (created by create_mlp2).
    - hiddens: The list of hidden layer sizes used in the encoder.
    - activation: Activation function to use in the decoder.
    - norm (str): Normalization type to use ('batch_norm' or 'layer_norm').
    - dropout (float): Dropout rate to apply in the decoder.
    - skip_connections (bool): Whether to include skip connections in the decoder.

    Returns:
    - autoencoder_model: A new model that includes both the encoder and the decoder.
    """

    # Get the input and representation (latent space) from the encoder
    encoder_input = encoder_model.input
    try:
        representation = encoder_model.get_layer('repr_layer').output
    except ValueError:
        # If 'repr_layer' is not found, use the last layer's output
        representation = encoder_model.layers[-1].output

    x = representation
    residual_layer = x
    has_residuals = skip_connections
    skipped_layers = encoder_model.skipped_layers if hasattr(encoder_model, 'skipped_layers') else 0

    # Reverse the hiddens list for the decoder
    decoder_hiddens = hiddens[::-1]

    # Build the decoder
    for i, units in enumerate(decoder_hiddens):
        x = Dense(units)(x)
        if norm == 'batch_norm':
            x = BatchNormalization()(x)
        x = activation(x) if callable(activation) else LeakyReLU()(x)

        # Add skip connections if enabled
        if has_residuals and skipped_layers > 0 and i % skipped_layers == 0 and i > 0:
            if x.shape[-1] != residual_layer.shape[-1]:
                residual_proj = Dense(x.shape[-1], use_bias=False)(residual_layer)
            else:
                residual_proj = residual_layer
            x = Add()([x, residual_proj])
            if norm == 'layer_norm':
                x = LayerNormalization()(x)
            if dropout > 0:
                x = Dropout(dropout)(x)
            residual_layer = x
        elif dropout > 0:
            x = Dropout(dropout)(x)

    # Output layer to reconstruct the input
    reconstructed = Dense(encoder_input.shape[-1], name='reconstructed')(x)

    # Create the autoencoder model
    autoencoder_model = Model(inputs=encoder_input, outputs=[representation, reconstructed])

    return autoencoder_model


# Create classification report tables for wandb
def create_metrics_table(y_true: np.ndarray, y_pred: np.ndarray, set_name: str) -> plt.Figure:
    """
    Creates a matplotlib figure containing classification metrics for model evaluation.

    Args:
        y_true (np.ndarray): Ground truth labels (class indices)
        y_pred (np.ndarray): Predicted labels (class indices) 
        set_name (str): Name of the dataset split ('Train' or 'Test')

    Returns:
        plt.Figure: Figure containing accuracy, per-class accuracy, precision, recall and F1 scores
    """
    # Calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate per-class accuracy
    class_accuracies = []
    for i in range(3):  # 3 classes
        mask = y_true == i
        class_accuracies.append(accuracy_score(y_true[mask], y_pred[mask]))

    # Calculate other metrics
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Define table data
    table_data = [
        ["Overall Accuracy", f"{accuracy:.3f}", "", "", ""],
        ["Class", "Accuracy", "Precision", "Recall", "F1-Score"],
        ["plus", f"{class_accuracies[2]:.3f}", f"{precision[2]:.3f}", f"{recall[2]:.3f}", f"{f1[2]:.3f}"],
        ["zero", f"{class_accuracies[1]:.3f}", f"{precision[1]:.3f}", f"{recall[1]:.3f}", f"{f1[1]:.3f}"],
        ["minus", f"{class_accuracies[0]:.3f}", f"{precision[0]:.3f}", f"{recall[0]:.3f}", f"{f1[0]:.3f}"]
    ]

    # Create table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Set title
    plt.title(f"{set_name} Classification Metrics", pad=20)

    return fig


def plot_posteriors(
        predictions: np.ndarray,
        y_delta: np.ndarray,
        lower_delta_threshold: float = -0.4,
        upper_delta_threshold: float = 0.4,
        suptitle: Optional[str] = "Posterior Probabilities vs. Delta"
) -> plt.Figure:
    """
    Plot the posterior probabilities for a 3-class combiner model: P(+|x), P(0|x), and P(-|x),
    each as a function of the delta (y_delta) on the same figure in four subplots.
    The fourth subplot shows the difference P(+|x) - P(-|x).

    Args:
        predictions (np.ndarray):
            A 2D array of shape (N, 3) representing the posterior probabilities for each sample.
            The columns should correspond to [P(+|x), P(0|x), P(-|x)] in that order.
        y_delta (np.ndarray):
            A 1D array of shape (N,) representing the delta values (Δ) for each sample.
        lower_delta_threshold (float, optional):
            The lower threshold for the delta values.
            Defaults to -0.4.
        upper_delta_threshold (float, optional):
            The upper threshold for the delta values.
            Defaults to 0.4.
        suptitle (str, optional):
            A custom super-title for the entire figure.
            Defaults to "Posterior Probabilities vs. Delta".

    Returns:
        matplotlib.figure.Figure:
            A Matplotlib Figure object containing the four subplots.
            Each subplot displays a scatter plot of delta vs. a posterior probability,
            with the last showing the difference P(+|x) - P(-|x).
    """
    # Ensure y_delta is 1D
    y_delta = np.squeeze(y_delta)
    if len(y_delta.shape) != 1:
        raise ValueError("y_delta must be a 1D array")

    # Create a figure with 4 subplots (2x2 grid), fixed figure size
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), sharey=False)

    # Create smoothing window
    window = 64
    
    # -------------------------------------------------
    # Subplot 1: P(+|x)
    # -------------------------------------------------
    axes[0,0].scatter(
        y_delta,
        predictions[:, 0],  # P(+|x) is assumed to be in column 0
        color='red',
        alpha=0.3,
        label='P(+|x)'
    )
    # Add smooth trend line using moving average
    sorted_idx = np.argsort(y_delta)
    x_smooth = y_delta[sorted_idx]
    y_smooth = predictions[:, 0][sorted_idx]
    y_smoothed = np.convolve(y_smooth, np.ones(window)/window, mode='valid')
    x_smoothed = x_smooth[window-1:]
    axes[0,0].plot(x_smoothed, y_smoothed, color='black', linewidth=0.9)
    # Add threshold lines
    axes[0,0].axvline(x=lower_delta_threshold, color='green', linestyle='--', alpha=0.5)
    axes[0,0].axvline(x=upper_delta_threshold, color='green', linestyle='--', alpha=0.5)
    
    axes[0,0].set_title("P(+|x)", fontsize=14)
    axes[0,0].set_xlabel("Delta", fontsize=12)
    axes[0,0].set_ylabel("Posterior Probability", fontsize=12)
    axes[0,0].set_xlim(-2.5, 2.5)
    axes[0,0].set_ylim(0, 1)

    # -------------------------------------------------
    # Subplot 2: P(-|x)
    # -------------------------------------------------
    axes[0,1].scatter(
        y_delta,
        predictions[:, 2],  # P(-|x) is assumed to be in column 2
        color='blue',
        alpha=0.3,
        label='P(-|x)'
    )
    # Add smooth trend line using moving average
    y_smooth = predictions[:, 2][sorted_idx]
    y_smoothed = np.convolve(y_smooth, np.ones(window)/window, mode='valid')
    axes[0,1].plot(x_smoothed, y_smoothed, color='black', linewidth=0.9)
    # Add threshold lines
    axes[0,1].axvline(x=lower_delta_threshold, color='green', linestyle='--', alpha=0.5)
    axes[0,1].axvline(x=upper_delta_threshold, color='green', linestyle='--', alpha=0.5)
    
    axes[0,1].set_title("P(-|x)", fontsize=14)
    axes[0,1].set_xlabel("Delta", fontsize=12)
    axes[0,1].set_ylabel("Posterior Probability", fontsize=12)
    axes[0,1].set_xlim(-2.5, 2.5)
    axes[0,1].set_ylim(0, 1)

    # -------------------------------------------------
    # Subplot 3: P(0|x)
    # -------------------------------------------------
    axes[1,0].scatter(
        y_delta,
        predictions[:, 1],  # P(0|x) is assumed to be in column 1
        color='gray',
        alpha=0.3,
        label='P(0|x)'
    )
    # Add smooth trend line using moving average
    y_smooth = predictions[:, 1][sorted_idx]
    y_smoothed = np.convolve(y_smooth, np.ones(window)/window, mode='valid')
    axes[1,0].plot(x_smoothed, y_smoothed, color='black', linewidth=0.9)
    # Add threshold lines
    axes[1,0].axvline(x=lower_delta_threshold, color='green', linestyle='--', alpha=0.5)
    axes[1,0].axvline(x=upper_delta_threshold, color='green', linestyle='--', alpha=0.5)
    
    axes[1,0].set_title("P(0|x)", fontsize=14)
    axes[1,0].set_xlabel("Delta", fontsize=12)
    axes[1,0].set_ylabel("Posterior Probability", fontsize=12)
    axes[1,0].set_xlim(-2.5, 2.5)
    axes[1,0].set_ylim(0, 1)

    # -------------------------------------------------
    # Subplot 4: P(+|x) - P(-|x)
    # -------------------------------------------------
    diff = predictions[:, 0] - predictions[:, 2]  # P(+|x) - P(-|x)
    axes[1,1].scatter(
        y_delta,
        diff,
        color='purple',
        alpha=0.3,
        label='P(+|x) - P(-|x)'
    )
    # Add smooth trend line using moving average
    y_smooth = diff[sorted_idx]
    y_smoothed = np.convolve(y_smooth, np.ones(window)/window, mode='valid')
    axes[1,1].plot(x_smoothed, y_smoothed, color='black', linewidth=0.9)
    # Add threshold lines
    axes[1,1].axvline(x=lower_delta_threshold, color='green', linestyle='--', alpha=0.5)
    axes[1,1].axvline(x=upper_delta_threshold, color='green', linestyle='--', alpha=0.5)
    
    axes[1,1].set_title("P(+|x) - P(-|x)", fontsize=14)
    axes[1,1].set_xlabel("Delta", fontsize=12)
    axes[1,1].set_ylabel("Probability Difference", fontsize=12)
    axes[1,1].set_xlim(-2.5, 2.5)
    axes[1,1].set_ylim(-1, 1)

    # Optional super-title
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)

    # Tighten up the layout
    plt.tight_layout()

    return fig


def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        xlabel: str = "Predicted",
        ylabel: str = "True",
        xticklabels: List[str] = None,
        yticklabels: List[str] = None
) -> plt.Figure:
    """
    Creates and returns a figure containing a confusion matrix plot.
    
    Args:
        y_true: Ground truth labels as 1D array
        y_pred: Predicted labels as 1D array  
        class_names: List of class names for axis labels
        title: Title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        xticklabels: Custom labels for x-axis ticks
        yticklabels: Custom labels for y-axis ticks
        
    Returns:
        matplotlib Figure object containing the confusion matrix plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Reverse the confusion matrix rows to flip the y-axis
    cm = np.flipud(cm)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use provided tick labels or default to class names
    xticks = xticklabels if xticklabels is not None else class_names
    yticks = yticklabels if yticklabels is not None else class_names[::-1]  # Reverse for y-axis

    # Plot confusion matrix using seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=xticks,
                yticklabels=yticks,
                ax=ax)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Adjust layout
    plt.tight_layout()

    return fig


def focal_loss(gamma: float = 3.0, alpha: float = 0.25):
    """
    Creates a focal loss function with specified gamma and alpha parameters.
    
    Args:
        gamma: Focusing parameter that modulates the rate at which easy examples are down-weighted
        alpha: Balancing parameter for class weights
        
    Returns:
        Callable focal loss function for model compilation
    """

    def focal_loss_fn(y_true, y_pred):
        # Scale predictions so that the class probabilities sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate focal loss
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy

        return tf.reduce_sum(loss, axis=-1)

    return focal_loss_fn


def create_mlp_moe(
        hiddens=None,
        combiner_hiddens=None,
        input_dim: int = 100,
        embed_dim: int = 128,
        skipped_layers: int = 1,
        skip_repr: bool = True,
        pretraining: bool = False,
        pretraining_paths: str = None,
        freeze_experts: bool = True,
        expert_paths: dict = None,
        mode: str = 'soft',  # 'soft' or 'hard'
        activation=None,
        norm: str = 'batch_norm',
        sam_rho: float = 1e-2,
        dropout: float = 0.2,
        name: str = 'mlp_moe',

) -> Model:
    """
    Create an MLP mixture of experts model with pre-trained experts and a combiner network.

    Parameters:
    - input_dim (int): The number of features in the input data.
    - hiddens (list): A list of integers where each integer is the number of units in a hidden layer.
    - skipped_layers (int): Number of layers between residual connections.
    - embed_dim (int): The number of features in the final representation vector.
    - skip_repr (bool): If True and skipped_layers > 0, adds a residual connection to the representation layer.
    - pretraining (bool): If True, the model will use PDS/PDC and have its representations normalized.
    - pretraining_paths (str): Path to the pre-trained model weights.
    - activation: Optional activation function to use. If None, defaults to LeakyReLU.
    - norm (str): Optional normalization type to use ('batch_norm' or 'layer_norm'). Default is 'batch_norm'.
    - sam_rho (float): Size of the neighborhood for perturbation in SAM. Default is 0.05. If 0.0, SAM is not used.
    - dropout (float): Dropout rate to apply after activations or residual connections. If 0.0, no dropout is applied.
    - expert_paths (dict): Dictionary containing paths to expert model weights:
        {
            'combiner': path to combiner model weights,
            'plus': path to plus expert weights,
            'zero': path to zero expert weights,
            'minus': path to minus expert weights
        }
    - combiner_hiddens (list): A list of integers for combiner network hidden layers. If None, uses same as hiddens.
    - freeze_experts (bool): If True, freeze the expert layers.
    - mode (str): Either 'soft' for weighted combination of experts or 'hard' for selecting single expert. Default is 'soft'.

    Returns:
    - Model: A Keras model instance.
    """
    if pretraining_paths is None:
        pretraining_paths = {
            'combiner': None,
            'plus': None,
            'zero': None,
            'minus': None
        }

    if hiddens is None:
        hiddens = MLP_HIDDENS

    if activation is None:
        activation = LeakyReLU()

    expert_output_dim = 1
    combiner_output_dim = 3

    input_layer = Input(shape=(input_dim,))

    # Create experts using the create_mlp function - each expert outputs a single regression value
    expert_plus = create_mlp(
        input_dim=input_dim,
        output_dim=expert_output_dim,
        hiddens=hiddens,
        skipped_layers=skipped_layers,
        embed_dim=embed_dim,
        skip_repr=skip_repr,
        pretraining=pretraining,
        activation=activation,
        norm=norm,
        sam_rho=sam_rho,
        dropout=dropout,
        name='expert_p'
    )

    expert_zero = create_mlp(
        input_dim=input_dim,
        output_dim=expert_output_dim,
        hiddens=hiddens,
        skipped_layers=skipped_layers,
        embed_dim=embed_dim,
        skip_repr=skip_repr,
        pretraining=pretraining,
        activation=activation,
        norm=norm,
        sam_rho=sam_rho,
        dropout=dropout,
        name='expert_nz'
    )

    expert_minus = create_mlp(
        input_dim=input_dim,
        output_dim=expert_output_dim,
        hiddens=hiddens,
        skipped_layers=skipped_layers,
        embed_dim=embed_dim,
        skip_repr=skip_repr,
        pretraining=pretraining,
        activation=activation,
        norm=norm,
        sam_rho=sam_rho,
        dropout=dropout,
        name='expert_m'
    )

    # Create combiner network - outputs class probabilities
    combiner = create_mlp(
        input_dim=input_dim,
        output_dim=combiner_output_dim,  # 3 classes: plus, mid, minus
        hiddens=combiner_hiddens if combiner_hiddens else hiddens,
        skipped_layers=skipped_layers,
        embed_dim=embed_dim,
        skip_repr=skip_repr,
        pretraining=pretraining,
        activation=activation,
        norm=norm,
        sam_rho=sam_rho,
        dropout=dropout,
        output_activation='softmax',  # Use softmax for class probabilities
        name='combiner'
    )

    # Load weights if paths are provided
    if expert_paths:
        if 'combiner' in expert_paths:
            combiner.load_weights(expert_paths['combiner'])
        if 'plus' in expert_paths:
            expert_plus.load_weights(expert_paths['plus'])
        if 'zero' in expert_paths:
            expert_zero.load_weights(expert_paths['zero'])
        if 'minus' in expert_paths:
            expert_minus.load_weights(expert_paths['minus'])

        if freeze_experts:
            for expert in [expert_plus, expert_zero, expert_minus]:
                for layer in expert.layers:
                    layer.trainable = False

    # Get expert outputs
    plus_output = expert_plus(input_layer)
    zero_output = expert_zero(input_layer)
    minus_output = expert_minus(input_layer)
    combiner_output = combiner(input_layer)

    # Extract routing probabilities
    routing_probs = combiner_output[1]  # Shape: (batch_size, 3)

    if mode == 'soft':
        # Soft mixture - weighted combination of experts
        plus_prob = Lambda(lambda x: x[:, PLUS_INDEX:PLUS_INDEX + 1])(routing_probs)
        zero_prob = Lambda(lambda x: x[:, MID_INDEX:MID_INDEX + 1])(routing_probs)
        minus_prob = Lambda(lambda x: x[:, MINUS_INDEX:MINUS_INDEX + 1])(routing_probs)

        plus_weighted = Multiply()([plus_output[1], plus_prob])
        zero_weighted = Multiply()([zero_output[1], zero_prob])
        minus_weighted = Multiply()([minus_output[1], minus_prob])
        forecast_head = Add(name='forecast_head')([plus_weighted, zero_weighted, minus_weighted])
    else:
        # Hard selection - choose expert with highest probability
        expert_selector = Lambda(lambda x: tf.argmax(x, axis=1))(routing_probs)
        forecast_head = Lambda(lambda args: tf.case({
            tf.equal(args[0], PLUS_INDEX): lambda: args[1],
            tf.equal(args[0], MID_INDEX): lambda: args[2],
            tf.equal(args[0], MINUS_INDEX): lambda: args[3]
        }, exclusive=True), name='forecast_head')(
            [expert_selector, plus_output[1], zero_output[1], minus_output[1]]
        )

    model = Model(inputs=input_layer, outputs=[routing_probs, forecast_head], name=name)
    return model


def create_hybrid_model(
        tsf_extractor: Model,
        mlp_input_dim: int = 23,
        mlp_hiddens=None,
        mlp_repr_dim: int = 9,
        final_hiddens=None,
        embed_dim: int = 10,
        output_dim: int = 1,
        pds: bool = False,
        l2_reg: float = None,
        dropout: float = 0.0,
        activation=None,
        norm: str = None,
        name: str = 'hybrid'
) -> Model:
    """
       Create a hybrid neural network model with a time series feature extraction branch (CNN or RNN)
       and an MLP branch.

       Parameters:
        - tsf_extractor (Model): A pre-built model for time series feature extraction (CNN or RNN).
        - mlp_input_dim (int): The number of features for the MLP branch.
        - output_dim (int): The dimension of the output layer.
        - embed_dim (int): The number of features in the final representation vector.
        - mlp_repr_dim (int): The number of features in the representation vector after the MLP branch.
        - mlp_hiddens (List[int]): List of integers for the MLP hidden layers.
        - final_hiddens (List[int]): List of integers for the hidden layers after concatenation.
        - pds (bool): If True, the model will be use PDS and there will have its representations normalized.
        - l2_reg (float): L2 regularization factor. Default is None (no regularization).
        - dropout (float): The fraction of the input units to drop. Default is 0.0 (no dropout).
        - activation: Optional activation function to use. If None, defaults to LeakyReLU.
        - norm (str): Optional normalization type to use ('batch_norm' or 'layer_norm'). Default is None (no normalization).
        - name (str): The name of the model. Default is 'hybrid'.

        Returns:
        - Model: A Keras model instance.
    """

    if final_hiddens is None:
        final_hiddens = [12]
    if mlp_hiddens is None:
        mlp_hiddens = [18]

    if activation is None:
        activation = LeakyReLU()

    # MLP Branch
    mlp_input = Input(shape=(mlp_input_dim,), name='mlp_input')
    x_mlp = mlp_input
    for units in mlp_hiddens:
        x_mlp = Dense(units, kernel_regularizer=l2(l2_reg) if l2_reg else None)(x_mlp)

        if norm == 'batch_norm':
            x_mlp = BatchNormalization()(x_mlp)
        elif norm == 'layer_norm':
            x_mlp = LayerNormalization()(x_mlp)

        if callable(activation):
            x_mlp = activation(x_mlp)
        else:
            x_mlp = LeakyReLU()(x_mlp)

        if dropout > 0.0:
            x_mlp = Dropout(dropout)(x_mlp)

    x_mlp = Dense(mlp_repr_dim, kernel_regularizer=l2(l2_reg) if l2_reg else None)(x_mlp)
    mlp_repr = activation(x_mlp) if callable(activation) else LeakyReLU()(x_mlp)

    # Concatenate the outputs of TSF Extractor and MLP branch
    concatenated = Concatenate()([tsf_extractor.output, mlp_repr])

    # Additional MLP Layer(s) after concatenation
    x_combined = concatenated
    for units in final_hiddens:
        x_combined = Dense(units, kernel_regularizer=l2(l2_reg) if l2_reg else None)(x_combined)

        if norm == 'batch_norm':
            x_combined = BatchNormalization()(x_combined)
        elif norm == 'layer_norm':
            x_combined = LayerNormalization()(x_combined)

        if callable(activation):
            x_combined = activation(x_combined)
        else:
            x_combined = LeakyReLU()(x_combined)

        if dropout > 0.0:
            x_combined = Dropout(dropout)(x_combined)

    # Final representation layer
    final_repr = Dense(embed_dim, kernel_regularizer=l2(l2_reg) if l2_reg else None)(x_combined)
    if pds:
        # Assuming NormalizeLayer is defined elsewhere for PDS normalization
        final_repr = NormalizeLayer(
            name='normalize_layer')(
            activation(final_repr) if callable(activation) else LeakyReLU()(final_repr))
    else:
        final_repr = activation(final_repr) if callable(activation) else LeakyReLU(name='final_repr_layer')(final_repr)

    if output_dim > 0:
        forecast_head = Dense(output_dim, activation='linear', name='forecast_head')(final_repr)
        model_output = [final_repr, forecast_head]
    else:
        model_output = final_repr

    # Create the model
    model = Model(inputs=[tsf_extractor.input, mlp_input], outputs=model_output, name=name)

    return model


def load_file_data(
        file_path: str,
        apply_log: bool = True,
        inputs_to_use: Optional[List[str]] = None,
        add_slope: bool = True,
        outputs_to_use: Optional[List[str]] = None,
        cme_speed_threshold: float = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes data from a single file.

    Parameters:
        - file_path (str): Path to the file.
        - apply_log (bool): Whether to apply a logarithmic transformation before normalization.
        - inputs_to_use (Optional[List[str]]): List of input types to include in the dataset.
        - add_slope (bool): If True, adds slope features to the dataset.
        - outputs_to_use (Optional[List[str]]): List of output types to include in the dataset. default is both ['p'] and ['delta_p']. Deprecated.
        - cme_speed_threshold (float): The threshold for CME speed. CMEs with speeds below (<) this threshold will be excluded. -1
        for no cmes

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Processed input data (X), target delta data (y), log proton intensity (logI), and log of p_t (logI_prev) as numpy arrays.
    """
    # Initialization and file reading
    if inputs_to_use is None:
        inputs_to_use = ['e0.5', 'e4.4', 'p6.1', 'p']
    if outputs_to_use is None:
        outputs_to_use = ['delta_p']

    # Dynamically define input columns based on inputs_to_use
    input_columns = []
    for input_type in inputs_to_use:
        input_columns += [f'{input_type}_tminus{i}' for i in range(24, 0, -1)] + [f'{input_type}_t']

    target_column = []
    # Dynamically define target column based on outputs_to_use

    if 'delta_p' in outputs_to_use:  # delta should come first
        target_column.append('delta_log_Intensity')
    if 'p' in outputs_to_use:
        target_column.append('Proton Intensity')

    cme_columns_to_zero_out = [
        'CME_DONKI_latitude', 'CME_DONKI_longitude', 'CME_DONKI_speed', 'CME_CDAW_MPA',
        'CME_CDAW_LinearSpeed', 'VlogV', 'DONKI_half_width', 'Accelaration',
        '2nd_order_speed_final', '2nd_order_speed_20R', 'CPA', 'Halo', 'Type2_Viz_Area',
        'solar_wind_speed', 'diffusive_shock', 'half_richardson_value'
    ]

    data = pd.read_csv(file_path)

    if cme_speed_threshold > -1:
        # Zero out CME columns for CMEs with speeds below the threshold
        data = zero_out_cme_below_threshold(data, cme_speed_threshold, cme_columns_to_zero_out)

    # Store log of p_t before any normalization
    logI_prev = np.log1p(data['p_t']) if apply_log else data['p_t']

    # Apply transformations and normalizations
    # Apply logarithmic transformation (if specified)
    if apply_log:
        data[input_columns] = np.log1p(data[input_columns])  # Adding 1 to avoid log(0)
        data['Proton Intensity'] = np.log1p(data['Proton Intensity'])  # Adding 1 to avoid log(0)

    # Normalize inputs between 0 and 1
    input_data = data[input_columns]
    input_data_normalized = (input_data - input_data.min()) / (input_data.max() - input_data.min())

    # Compute and add slopes (if specified)
    if add_slope:
        for input_type in inputs_to_use:
            # print(f"Computing slopes for {input_type}...")
            slope_column_names = generate_slope_column_names([input_type])  # Generate for current input type
            slope_values = compute_slope(input_data_normalized, input_type)
            for slope_column, slope_index in zip(slope_column_names, range(slope_values.shape[1])):
                # print(f"Adding {slope_column} to input cme_files...")
                input_data_normalized[slope_column] = slope_values[:, slope_index]
            # input_columns.extend(slope_column_names)

    # print the columns one by one
    # for col in input_data_normalized.columns:
    #     print(col)
    # order - e0.5, e1.8, p, e0.5 slope, e1.8 slope, p slope
    target_data = data[target_column]

    # Get log proton intensity
    logI = data['Proton Intensity'].values

    # Get delta log intensity target
    y = target_data.values

    if cme_speed_threshold > -1:
        # Process and append CME features
        cme_features = preprocess_cme_features(data, inputs_to_use)
        combined_input = pd.concat([input_data_normalized, cme_features], axis=1)
        X = combined_input.values
    else:
        X = input_data_normalized.values.reshape((input_data_normalized.shape[0], -1, 1))

    # Return processed X, y, logI and logI_prev
    return X, y, logI, logI_prev


def stratified_split(
        X: np.ndarray,
        y: np.ndarray,
        seed: int = None,
        shuffle: bool = True,
        debug: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset into subtraining and validation sets using stratified sampling.
    The validation is a quarter of the dataset, and the rest is used for subtraining.

    Parameters:
    X (np.ndarray): Feature matrix of shape (n_samples, n_features).
    y (np.ndarray): Label vector of shape (n_samples, 1).
    shuffle (bool): Whether to shuffle the data before splitting. Default is True.
    seed (int): Random seed for reproducibility. Default is None.
    debug (bool): Whether to plot the distributions of the original, subtrain, and validation sets. Default is False.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Split feature and label matrices:
        - X_subtrain: Features for the subtraining set.
        - y_subtrain: Labels for the subtraining set.
        - X_val: Features for the validation set.
        - y_val: Labels for the validation set.
    """
    if shuffle: np.random.seed(seed)
    # Sort the data by the labels
    sorted_indices = np.argsort(y, axis=0).flatten()
    X_sorted, y_sorted = X[sorted_indices], y[sorted_indices]
    # Calculate the number of validation samples
    num_samples = X.shape[0]
    # val_size = int(num_samples * split)
    # Initialize lists to hold subtraining and validation data
    X_subtrain, y_subtrain, X_val, y_val = [], [], [], []
    # Divide into groups of 4 and split into subtrain and validation
    for i in range(0, num_samples, 4):
        group_indices = list(range(i, min(i + 4, num_samples)))
        if shuffle: np.random.shuffle(group_indices)  # Shuffle within the group
        val_indices = group_indices[:1]
        subtrain_indices = group_indices[1:]
        # Append the samples to the subtraining and validation sets
        X_val.extend(X_sorted[val_indices])
        y_val.extend(y_sorted[val_indices])
        X_subtrain.extend(X_sorted[subtrain_indices])
        y_subtrain.extend(y_sorted[subtrain_indices])

    # Convert lists back to arrays
    X_subtrain, y_subtrain = np.array(X_subtrain), np.array(y_subtrain)
    X_val, y_val = np.array(X_val), np.array(y_val)

    # Ensure the largest y is in the validation set
    max_y_index = np.argmax(y_sorted)  # Index of the largest y value
    max_y_val = y_sorted[max_y_index]  # Largest y value
    # Check if the largest y value is not in the validation set
    if max_y_val not in y_val:
        # Add the sample with the largest y value to the validation set
        X_val = np.vstack([X_val, X_sorted[max_y_index].reshape(1, -1)])
        y_val = np.vstack([y_val, max_y_val.reshape(1, -1)])
        # # Remove the largest y from the subtraining set
        # mask = y_subtrain != max_y_val
        # X_subtrain = X_subtrain[mask.flatten()]
        # y_subtrain = y_subtrain[mask.flatten()]

    if debug:
        plot_distributions(y, y_subtrain, y_val)

    return X_subtrain, y_subtrain, X_val, y_val


def stratified_groups(y: np.ndarray, batch_size: int, debug: bool = False) -> np.ndarray:
    """
    Create stratified groups from the dataset by sorting it based on the labels.
    The number of groups corresponds to the batch size, and each group will have
    samples with similar label distributions. The result is a 2D array where each
    group is padded to the same size. Verified to work.

    Parameters:
    -----------
    y : np.ndarray
        Label vector of shape (n_samples, 1).
    batch_size : int
        Number of groups, which will correspond to the number of samples in each batch.

    Returns:
    --------
    np.ndarray:
        A 2D array where each row represents a stratified group, and all rows have the same length.
    """
    # Sort the dataset along the second dimension (axis=0)
    sorted_indices = np.argsort(y, axis=0).flatten()

    # Debugging: Check the shape of y and print sorted indices
    if debug:
        print(f"Shape of y: {y.shape}")
        print(f"Unique values in y: {np.unique(y)}")
        print(f"Sorted indices: {sorted_indices}")
        print(f"Labels sorted by indices: {y[sorted_indices].flatten()}")  # To see how the labels are sorted

    # Create groups by slicing the sorted data indices
    groups = np.array_split(sorted_indices, batch_size)

    # Find the maximum group size
    max_size = max(len(group) for group in groups)

    # Pad the groups with their last element to make all groups the same size
    padded_groups = np.array([
        np.pad(group, (0, max_size - len(group)), 'edge') for group in groups
    ])

    return padded_groups


# def stratified_groups(y: np.ndarray, batch_size: int, debug: bool = False) -> np.ndarray:
#     """
#     Create stratified groups from the dataset by sorting it based on the labels.
#     The number of groups corresponds to the batch size per GPU.
#
#     Parameters:
#     -----------
#     y : np.ndarray
#         Label vector of shape (n_samples, 1).
#     batch_size : int
#         Number of samples in each batch per GPU.
#
#     Returns:
#     --------
#     np.ndarray:
#         A 2D array where each row represents a stratified group, and all rows have the same length.
#     """
#     # Sort the dataset along the second dimension (axis=0)
#     sorted_indices = np.argsort(y, axis=0).flatten()
#
#     # Create groups by slicing the sorted data indices
#     groups = np.array_split(sorted_indices, batch_size)
#
#     # Find the maximum group size
#     max_size = max(len(group) for group in groups)
#
#     # Pad the groups with their last element to make all groups the same size
#     padded_groups = np.array([
#         np.pad(group, (0, max_size - len(group)), 'edge') for group in groups
#     ])
#
#     return padded_groups


def stratified_data_generator(
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,  # Assuming groups is a 2D array
        shuffle: bool = True,
        debug: bool = False
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generator that yields stratified batches of (X, y) by selecting one sample from each group.
    The groups are passed as a 2D array of sample indices.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Label vector of shape (n_samples,).
    groups : np.ndarray
        Precomputed groups of sample indices for stratified sampling, shape (n_groups, group_size).
    shuffle : bool, optional
        If True, shuffles the groups and the elements within each group before each epoch (default is True).
    debug : bool, optional
        If True, prints the generated batches for debugging purposes (default is True).

    Yields:
    -------
    Tuple[np.ndarray, np.ndarray]:
        Batches of feature matrix and label vector of size (batch_size, n_features) and (batch_size,) respectively.
    """

    while True:
        if shuffle: np.apply_along_axis(np.random.shuffle, 1, groups)  # Shuffle within each group (columns)
        # Select the first element from each group to form the batch
        batch_indices = groups[:, 0]

        # Optionally, shuffle the order of the selected samples to randomize batch order
        np.random.shuffle(batch_indices)

        # Create the feature and label batches using the selected indices
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]

        # Ensure the labels have the correct shape
        batch_y = batch_y.reshape(-1)

        # Debugging: Print the current batch
        if debug:
            print(f'Batch shape: {batch_X.shape}, {batch_y.shape}')
            print(f"Batch indices: {batch_indices}")
            # print(f"Batch X:\n{batch_X}")
            print(f"Batch y:\n{batch_y}")

        # Yield the current batch
        yield batch_X, batch_y


def stratified_data_generator_cls(
        X: np.ndarray,
        y_labels: np.ndarray,
        delta: np.ndarray,
        groups: np.ndarray,
        shuffle: bool = True,
        debug: bool = False
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generator that yields stratified batches of (X, y_labels_concat) by selecting one sample 
    from each group in `groups`. The groups are precomputed as a 2D array of sample indices.

    This version differs from the original in that it returns delta values along with 
    the features and labels, which can be useful for handling imbalanced datasets.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y_labels : np.ndarray
        Label matrix of shape (n_samples, n_classes) or (n_samples,) depending on the use case.
    delta : np.ndarray
        Delta values vector of shape (n_samples,).
    groups : np.ndarray
        Precomputed groups of sample indices, shape (n_groups, group_size), obtained from stratified 
        grouping functions.
    shuffle : bool, optional
        If True, shuffles the groups and the elements within each group before each epoch (default is True).
    debug : bool, optional
        If True, prints the generated batches for debugging purposes (default is False).

    Yields
    ------
    Tuple[np.ndarray, np.ndarray]
        A tuple `(batch_X, batch_y_concat)` where:
        - `batch_X` is a feature batch of shape (batch_size, n_features).
        - `batch_y_concat` is the concatenation of `y_labels` and `delta` with shape 
          (batch_size, n_classes + 1).
    """
    while True:
        # Shuffle within each group if requested
        if shuffle:
            np.apply_along_axis(np.random.shuffle, 1, groups)

        # Select the first element from each group to form the batch
        batch_indices = groups[:, 0]

        # Optionally, shuffle the order of the selected samples to randomize batch order
        if shuffle:
            np.random.shuffle(batch_indices)

        # Create the feature, label and delta batches using the selected indices
        batch_X = X[batch_indices]
        batch_y = y_labels[batch_indices]
        batch_delta = delta[batch_indices]

        # Concatenate y_labels and delta
        batch_y_concat = np.concatenate([batch_y, batch_delta[:, None]], axis=1)

        # Debugging: Print the batch details if required
        if debug:
            print(f"Batch shape: X={batch_X.shape}, y_concat={batch_y_concat.shape}")
            print(f"Batch indices: {batch_indices}")

        # Yield the current batch
        yield batch_X, batch_y_concat


def stratified_data_generator_cls2(
        X: np.ndarray,
        y_labels: np.ndarray,
        delta: np.ndarray,
        groups: np.ndarray,
        shuffle: bool = True,
        debug: bool = False
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Generator that yields stratified batches of (X, y_labels, delta) by selecting one sample 
    from each group in `groups`. The groups are precomputed as a 2D array of sample indices.

    This version differs from the original in that it returns delta values along with 
    the features and labels, which can be useful for handling imbalanced datasets.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y_labels : np.ndarray
        Label matrix of shape (n_samples, n_classes) or (n_samples,) depending on the use case.
    delta : np.ndarray
        Delta values vector of shape (n_samples,).
    groups : np.ndarray
        Precomputed groups of sample indices, shape (n_groups, group_size), obtained from stratified 
        grouping functions.
    shuffle : bool, optional
        If True, shuffles the groups and the elements within each group before each epoch (default is True).
    debug : bool, optional
        If True, prints the generated batches for debugging purposes (default is False).

    Yields
    ------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple `(batch_X, batch_y, batch_delta)` where:
        - `batch_X` is a feature batch of shape (batch_size, n_features).
        - `batch_y` is the labels batch of shape (batch_size, n_classes).
        - `batch_delta` is the delta values batch of shape (batch_size,).
    """
    while True:
        # Shuffle within each group if requested
        if shuffle:
            np.apply_along_axis(np.random.shuffle, 1, groups)

        # Select the first element from each group to form the batch
        batch_indices = groups[:, 0]

        # Optionally, shuffle the order of the selected samples to randomize batch order
        if shuffle:
            np.random.shuffle(batch_indices)

        # Create the feature, label and delta batches using the selected indices
        batch_X = X[batch_indices]
        batch_y = y_labels[batch_indices]
        batch_delta = delta[batch_indices].reshape(-1)  # Ensure delta is 1D

        # Debugging: Print the batch details if required
        if debug:
            print(f"Batch shape: X={batch_X.shape}, y={batch_y.shape}, delta={batch_delta.shape}")
            print(f"Batch indices: {batch_indices}")

        # Yield the current batch
        yield batch_X, batch_y, batch_delta


# def stratified_data_generator(
#         X: np.ndarray,
#         y: np.ndarray,
#         groups: np.ndarray,
#         global_batch_size: int,
#         shuffle: bool = True,
#         debug: bool = False
# ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
#     """
#     Generator that yields stratified batches of (X, y) for MirroredStrategy.
#
#     Parameters:
#     -----------
#     X : np.ndarray
#         Feature matrix of shape (n_samples, n_features).
#     y : np.ndarray
#         Label vector of shape (n_samples,).
#     groups : np.ndarray
#         Precomputed groups of sample indices for stratified sampling.
#     global_batch_size : int
#         Total batch size across all GPUs.
#     shuffle : bool, optional
#         If True, shuffles the groups and the elements within each group before each epoch (default is True).
#     debug : bool, optional
#         If True, prints the generated batches for debugging purposes (default is False).
#
#     Yields:
#     -------
#     Tuple[np.ndarray, np.ndarray]:
#         Batches of feature matrix and label vector of size (global_batch_size, n_features) and (global_batch_size,) respectively.
#     """
#     while True:
#         if shuffle:
#             np.apply_along_axis(np.random.shuffle, 1, groups)
#
#         # Select samples to form the global batch
#         batch_indices = groups[:, :global_batch_size // len(groups)].flatten()
#
#         # Optionally, shuffle the order of the selected samples
#         if shuffle:
#             np.random.shuffle(batch_indices)
#
#         # Create the feature and label batches using the selected indices
#         batch_X = X[batch_indices]
#         batch_y = y[batch_indices]
#
#         # Ensure the labels have the correct shape
#         batch_y = batch_y.reshape(-1)
#
#         if debug:
#             print(f'Batch shape: {batch_X.shape}, {batch_y.shape}')
#             print(f"Batch y:\n{batch_y}")
#
#         yield batch_X, batch_y


def stratified_batch_dataset(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True
) -> Tuple[tf.data.Dataset, int]:
    """
    Creates a TensorFlow dataset from the stratified data generator, with groups generated only once.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Label vector of shape (n_samples,).
    batch_size : int
        Number of samples in each batch.
    shuffle : bool, optional
        If True, shuffles the groups and the elements within each group before each epoch (default is True).

    Returns:
    --------
    Tuple[tf.data.Dataset, int]:
        - A TensorFlow dataset object with stratified batches.
        - The number of steps per epoch (i.e., how many batches per epoch).
    """
    # Generate the stratified groups once
    groups = stratified_groups(y, batch_size)

    # Reshape X to remove the extra dimension if it exists
    # if len(X.shape) == 3:
    #     X = X.reshape(X.shape[0], -1)

    # Use from_generator to create a dataset from the stratified_data_generator
    dataset = tf.data.Dataset.from_generator(
        lambda: stratified_data_generator(X, y, groups, shuffle=shuffle),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, X.shape[1]), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.float32)
        )
    )
    # Compute the number of steps per epoch
    steps_per_epoch = len(y) // batch_size
    # Prefetch for performance optimization
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, steps_per_epoch


def stratified_batch_dataset_cls(
        X: np.ndarray,
        y_labels: np.ndarray,
        delta: np.ndarray,
        batch_size: int,
        shuffle: bool = True
) -> Tuple[tf.data.Dataset, int]:
    """
    Creates a TensorFlow dataset from the stratified data generator for a classification scenario,
    leveraging a continuous label array `delta` to stratify and a separate array `y_labels` for 
    the final training labels.

    This function uses `delta` solely for determining how to group samples into batches. These 
    batches are formed by slicing the dataset after sorting by `delta`, effectively grouping 
    together samples with similar continuous label values. The `y_labels` (e.g., one-hot encoded 
    classes) are the actual training targets.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y_labels : np.ndarray
        Label matrix of shape (n_samples, n_classes) or (n_samples,) depending on the use case.
        For classification, this could be one-hot encoded targets.
    delta : np.ndarray
        Continuous label vector of shape (n_samples,). Used only for stratification.
        Must be numeric and sortable.
    batch_size : int
        Number of samples in each batch.
    shuffle : bool, optional
        If True, shuffles the groups and the elements within each group before each epoch
        (default is True).

    Returns
    -------
    Tuple[tf.data.Dataset, int]
        A tuple of:
        - A TensorFlow dataset object that yields `(features, combined_labels)` tuples where
          combined_labels concatenates y_labels and delta.
        - An integer representing the number of steps (batches) per epoch.

    Notes
    -----
    - The dataset is created using `tf.data.Dataset.from_generator`, ensuring compatibility 
      with large datasets and on-demand batch construction.
    - Prefetching is enabled for better performance.
    - Ensure that `X`, `delta`, and `y_labels` have the same first dimension (n_samples).
    """
    # Generate stratified groups once using the continuous labels
    groups = stratified_groups(delta, batch_size)

    # The generator will yield batches of (X_batch, y_labels_batch, delta_batch)
    dataset = tf.data.Dataset.from_generator(
        lambda: stratified_data_generator_cls(
            X, y_labels, delta, groups, shuffle=shuffle,
        ),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, X.shape[1]), dtype=tf.float32),  # X_batch
            tf.TensorSpec(shape=(batch_size, y_labels.shape[1] + 1), dtype=tf.float32)  # Combined y_labels and delta
        )
    )

    # Compute the number of steps per epoch. Integer division used since partial batch is discarded.
    steps_per_epoch = len(delta) // batch_size

    # Prefetch the dataset for performance optimization
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, steps_per_epoch


def stratified_batch_dataset_cls2(
        X: np.ndarray,
        y_labels: np.ndarray,
        delta: np.ndarray,
        batch_size: int,
        shuffle: bool = True
) -> Tuple[tf.data.Dataset, int]:
    """
    Creates a TensorFlow dataset from the stratified data generator for a classification scenario,
    leveraging a continuous label array `delta` to stratify and a separate array `y_labels` for 
    the final training labels.

    This function uses `delta` solely for determining how to group samples into batches. These 
    batches are formed by slicing the dataset after sorting by `delta`, effectively grouping 
    together samples with similar continuous label values. The `y_labels` (e.g., one-hot encoded 
    classes) are the actual training targets.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y_labels : np.ndarray
        Label matrix of shape (n_samples, n_classes) or (n_samples,) depending on the use case.
        For classification, this could be one-hot encoded targets.
    delta : np.ndarray
        Continuous label vector of shape (n_samples,). Used only for stratification.
        Must be numeric and sortable.
    batch_size : int
        Number of samples in each batch.
    shuffle : bool, optional
        If True, shuffles the groups and the elements within each group before each epoch
        (default is True).

    Returns
    -------
    Tuple[tf.data.Dataset, int]
        A tuple of:
        - A TensorFlow dataset object that yields `(features, y_labels, delta)` tuples.
        - An integer representing the number of steps (batches) per epoch.

    Notes
    -----
    - The dataset is created using `tf.data.Dataset.from_generator`, ensuring compatibility 
      with large datasets and on-demand batch construction.
    - Prefetching is enabled for better performance.
    - Ensure that `X`, `delta`, and `y_labels` have the same first dimension (n_samples).
    """
    # Generate stratified groups once using the continuous labels
    groups = stratified_groups(delta, batch_size)

    # The generator will yield batches of (X_batch, y_labels_batch, delta_batch)
    dataset = tf.data.Dataset.from_generator(
        lambda: stratified_data_generator_cls2(
            X, y_labels, delta, groups, shuffle=shuffle,
        ),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, X.shape[1]), dtype=tf.float32),  # X_batch
            tf.TensorSpec(shape=(batch_size, y_labels.shape[1]), dtype=tf.float32),  # y_labels_batch
            tf.TensorSpec(shape=(batch_size,), dtype=tf.float32)  # delta_batch
        )
    )

    # Compute the number of steps per epoch. Integer division used since partial batch is discarded.
    steps_per_epoch = len(delta) // batch_size

    # Prefetch the dataset for performance optimization
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, steps_per_epoch


# def stratified_batch_dataset(
#         X: np.ndarray,
#         y: np.ndarray,
#         batch_size: int,
#         num_replicas: int = 1,
#         shuffle: bool = True
# ) -> Tuple[tf.data.Dataset, int]:
#     """
#     Creates a TensorFlow dataset with stratified batches for MirroredStrategy.
#
#     Parameters:
#     -----------
#     X : np.ndarray
#         Feature matrix of shape (n_samples, n_features).
#     y : np.ndarray
#         Label vector of shape (n_samples,).
#     batch_size : int
#         Number of samples in each batch per GPU.
#     num_replicas : int
#         Number of GPUs (default is 4).
#     shuffle : bool, optional
#         If True, shuffles the groups and the elements within each group before each epoch (default is True).
#
#     Returns:
#     --------
#     Tuple[tf.data.Dataset, int]:
#         - A TensorFlow dataset object with stratified batches.
#         - The number of steps per epoch (i.e., how many batches per epoch).
#     """
#     global_batch_size = batch_size * num_replicas
#     # Generate the stratified groups once
#     groups = stratified_groups(y, batch_size)
#     # Use from_generator to create a dataset from the stratified_data_generator
#     dataset = tf.data.Dataset.from_generator(
#         lambda: stratified_data_generator(X, y, groups, global_batch_size, shuffle=shuffle),
#         output_signature=(
#             tf.TensorSpec(shape=(global_batch_size, X.shape[1]), dtype=tf.float32),
#             tf.TensorSpec(shape=(global_batch_size,), dtype=tf.float32)
#         )
#     )
#     # Compute the number of steps per epoch
#     steps_per_epoch = len(y) // global_batch_size
#     # Prefetch for performance optimization
#     dataset = dataset.prefetch(tf.data.AUTOTUNE)
#
#     return dataset, steps_per_epoch


def load_stratified_folds(
        fold_dir_path: str,
        inputs_to_use: list,
        add_slope: bool,
        outputs_to_use: list,
        cme_speed_threshold: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
        debug: bool = False
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Loads pre-split stratified folds from a directory structure using build_dataset function
    and yields one fold at a time. Each fold directory should contain 'subtraining' and
    'validation' subdirectories with CSV files.

    Directory structure expected:
    fold_dir_path/
        fold0/
            subtraining/
                *.csv files
            validation/
                *.csv files
        fold1/
            ...
        fold2/
            ...
        fold3/
            ...

    Parameters:
        fold_dir_path (str): Path to the directory containing the fold directories
        inputs_to_use (list): List of input features to use from the CSV files
        add_slope (bool): Whether to add slope features
        outputs_to_use (list): List of output variables to use from the CSV files
        cme_speed_threshold (int): Threshold for CME speed filtering
        shuffle (bool): Whether to shuffle the data after loading. Default is True.
        seed (Optional[int]): Random seed for reproducibility. Default is None.
        debug (bool): Whether to enable debug features. Default is False.

    Yields:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Split feature and label matrices:
            - X_subtrain: Features for the subtraining set
            - y_subtrain: Labels for the subtraining set
            - X_val: Features for the validation set
            - y_val: Labels for the validation set

    Raises:
        FileNotFoundError: If any expected fold directory is missing
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    fold_dir = Path(fold_dir_path)

    # Verify directory structure
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

    # Process each fold
    for fold_idx in range(4):
        current_fold = fold_dir / f"fold{fold_idx}"

        if not current_fold.exists():
            raise FileNotFoundError(f"Missing fold directory: {current_fold}")

        # Load subtraining data using build_dataset
        subtrain_dir = str(current_fold / "subtraining")
        X_subtrain, y_subtrain, _, _ = build_dataset(
            subtrain_dir,
            inputs_to_use=inputs_to_use,
            add_slope=add_slope,
            outputs_to_use=outputs_to_use,
            cme_speed_threshold=cme_speed_threshold,
            shuffle_data=shuffle
        )

        # Load validation data using build_dataset
        val_dir = str(current_fold / "validation")
        X_val, y_val, _, _ = build_dataset(
            val_dir,
            inputs_to_use=inputs_to_use,
            add_slope=add_slope,
            outputs_to_use=outputs_to_use,
            cme_speed_threshold=cme_speed_threshold,
            shuffle_data=shuffle
        )

        if debug:
            print(f"Fold {fold_idx}:")
            print(f"Subtraining shapes: X={X_subtrain.shape}, y={y_subtrain.shape}")
            print(f"Validation shapes: X={X_val.shape}, y={y_val.shape}")
            # Could add distribution plots or other debug info here if needed

        yield X_subtrain, y_subtrain, X_val, y_val


def stratified_4fold_split(
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True,
        seed: int = None,
        debug: bool = False
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Splits the dataset into 4 folds using stratified sampling and yields one fold at a time.
    This method ensures no overlap between validation folds.

    Parameters:
    X (np.ndarray): Feature matrix of shape (n_samples, n_features).
    y (np.ndarray): Label vector of shape (n_samples, 1).
    shuffle (bool): Whether to shuffle the data after splitting. Default is True.
    seed (int): Random seed for reproducibility. Default is None.
    debug (bool): Whether to plot the distributions of the original, subtrain, and validation sets. Default is False.

    Yields:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Split feature and label matrices:
        - X_subtrain: Features for the subtraining set.
        - y_subtrain: Labels for the subtraining set.
        - X_val: Features for the validation set.
        - y_val: Labels for the validation set.
    """
    if shuffle:
        np.random.seed(seed)

    # Sort the data by the labels
    sorted_indices = np.argsort(y, axis=0).flatten()
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]

    num_samples = X.shape[0]
    group_size = 4  # Size of each group
    num_groups = num_samples // group_size

    # Initialize lists to hold indices for each fold
    fold_indices = [[] for _ in range(4)]

    # Divide the sorted data into groups of 4 and assign indices to each fold
    for group_start in range(0, num_groups * group_size, group_size):
        group_indices = list(range(group_start, group_start + group_size))

        # Ensure that each fold gets one unique sample from each group
        for fold in range(4):
            val_index = group_indices[fold]
            fold_indices[fold].append(val_index)

    # Generate the folds
    for fold in range(4):
        val_indices = fold_indices[fold]
        subtrain_indices = [index for i in range(4) if i != fold for index in fold_indices[i]]

        X_val = X_sorted[val_indices]
        y_val = y_sorted[val_indices]
        X_subtrain = X_sorted[subtrain_indices]
        y_subtrain = y_sorted[subtrain_indices]

        # Optional: shuffle the resulting folds
        if shuffle:
            perm = np.random.permutation(len(X_subtrain))
            X_subtrain, y_subtrain = X_subtrain[perm], y_subtrain[perm]

            perm = np.random.permutation(len(X_val))
            X_val, y_val = X_val[perm], y_val[perm]

        if debug:
            plot_distributions(y, y_subtrain, y_val)

        yield X_subtrain, y_subtrain, X_val, y_val


def plot_distributions(y_train: np.ndarray, y_subtrain: np.ndarray, y_val: np.ndarray) -> None:
    """
    Plots the distributions of the original training set, subtraining set, and validation set.
    Also prints the min, max, and mode values of the labels for each set.

    Parameters:
    y_train (np.ndarray): Labels of the original training set.
    y_subtrain (np.ndarray): Labels of the subtraining set.
    y_val (np.ndarray): Labels of the validation set.
    """

    def print_stats(y, name):
        min_y = np.min(y)
        max_y = np.max(y)
        mode_y = stats.mode(y).mode[0]
        print(f"{name} - Min: {min_y}, Max: {max_y}, Mode: {mode_y}")

    print_stats(y_train, "Train")
    print_stats(y_subtrain, "Subtrain")
    print_stats(y_val, "Validation")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(y_train, bins=50, color='blue', alpha=0.7, label='Train')
    plt.title('Original Training Set Distribution')

    plt.subplot(1, 3, 2)
    plt.hist(y_subtrain, bins=50, color='green', alpha=0.7, label='Subtrain')
    plt.title('Subtraining Set Distribution')

    plt.subplot(1, 3, 3)
    plt.hist(y_val, bins=50, color='red', alpha=0.7, label='Validation')
    plt.title('Validation Set Distribution')

    plt.show()


def build_dataset(
        directory_path: str,
        shuffle_data: bool = False,
        apply_log: bool = True,
        inputs_to_use: Optional[List[str]] = None,
        add_slope: bool = True,
        outputs_to_use: Optional[List[str]] = None,
        cme_speed_threshold: float = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds a dataset by processing files in a given directory.

    Reads SEP event files from the specified directory, processes them to extract
    input and target cme_files, normalizes the values between 0 and 1 for the columns
    of interest, excludes rows where proton intensity is -9999, and optionally shuffles the cme_files.

     Parameters:
        - directory_path (str): Path to the directory containing the sep_event_X files.
        - shuffle_data (bool): If True, shuffle the cme_files before returning.
        - apply_log (bool): Whether to apply a logarithmic transformation before normalization.
        - inputs_to_use (List[str]): List of input types to include in the dataset. Default is ['e0.5', 'e1.8', 'p'].
        - add_slope (bool): If True, adds slope features to the dataset.
        - outputs_to_use (List[str]): List of output types to include in the dataset. Default is both ['p'] and ['delta_p'].
        - cme_speed_threshold (float): The threshold for CME speed. CMEs with speeds below (<) this threshold will be excluded. -1

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the combined input data (X), target data (y), log proton intensity (logI), and previous log proton intensity (logI_prev).
    """
    all_inputs, all_targets, all_logI, all_logI_prev = [], [], [], []

    for file_name in os.listdir(directory_path):
        if file_name.endswith('_ie_trim.csv'):
            file_path = os.path.join(directory_path, file_name)
            X, y, logI, logI_prev = load_file_data(
                file_path,
                apply_log,
                inputs_to_use,
                add_slope,
                outputs_to_use,
                cme_speed_threshold)
            all_inputs.append(X)
            all_targets.append(y)
            all_logI.append(logI)
            all_logI_prev.append(logI_prev)

    X_combined = np.vstack(all_inputs)
    y_combined = np.concatenate(all_targets)
    logI_combined = np.concatenate(all_logI)
    logI_prev_combined = np.concatenate(all_logI_prev)

    if shuffle_data:
        X_combined, y_combined, logI_combined, logI_prev_combined = shuffle(
            X_combined, y_combined, logI_combined, logI_prev_combined,
            random_state=seed_value
        )

    return X_combined, y_combined, logI_combined, logI_prev_combined


def locate_high_deltas(
        directory_path: str,
        shuffle_data: bool = True,
        apply_log: bool = True,
        inputs_to_use: Optional[List[str]] = None,
        add_slope: bool = True,
        outputs_to_use: Optional[List[str]] = None,
        cme_speed_threshold: float = -1
) -> None:
    """
    Builds a dataset by processing files in a given directory.

    Reads SEP event files from the specified directory, processes them to extract
    input and target cme_files, normalizes the values between 0 and 1 for the columns
    of interest, excludes rows where proton intensity is -9999, and optionally shuffles the cme_files.

     Parameters:
        - directory_path (str): Path to the directory containing the sep_event_X files.
        - shuffle_data (bool): If True, shuffle the cme_files before returning.
        - apply_log (bool): Whether to apply a logarithmic transformation before normalization.
        - inputs_to_use (List[str]): List of input types to include in the dataset. Default is ['e0.5', 'e1.8', 'p'].
        - add_slope (bool): If True, adds slope features to the dataset.
        - outputs_to_use (List[str]): List of output types to include in the dataset. Default is both ['p'] and ['delta_p'].
        - cme_speed_threshold (float): The threshold for CME speed. CMEs with speeds below (<) this threshold will be excluded. -1

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing the combined input data (X) and target data (y).
    """

    for file_name in os.listdir(directory_path):
        print(f'Processing file: {file_name}')
        if file_name.endswith('_ie_trim.csv'):
            file_path = os.path.join(directory_path, file_name)
            X, y = load_file_data(
                file_path,
                apply_log,
                inputs_to_use,
                add_slope,
                outputs_to_use,
                cme_speed_threshold)

            # Counting labels
            label_count_above_2 = np.sum(y > 2)
            label_count_above_1_and_below_minus1 = np.sum((y > 1) | (y < -1))

            print(f'Count of labels above 2: {label_count_above_2}')
            print(f'Count of labels above 1 and below -1: {label_count_above_1_and_below_minus1}')


def generate_feature_names(inputs_to_use: List[str], add_slope: bool) -> List[str]:
    """
    Generates a list of feature names based on specified input types and whether slope features should be added.

    This function dynamically creates feature names for time-lagged inputs and optionally for slope features,
    given a list of input types (e.g., ['e0.5', 'e1.8', 'p']). For each input type, it generates names for features
    representing lagged values up to 24 hours prior ('tminus24' to 't') and, if requested, adds names for the slope
    feature associated with each input type, indicating specific time lags between which the slope is calculated.

    Parameters:
    - inputs_to_use (List[str]): A list of strings representing the input types for which features should be generated.
    - add_slope (bool): A flag indicating whether to include slope features for each input type in the output list.

    Returns:
    - List[str]: A list containing the generated feature names.

    Example:
     generate_feature_names(['e0.5', 'p'], add_slope=True)
    ['e0.5_tminus24', 'e0.5_tminus23', ..., 'e0.5_t', 'e0.5_slope_tminus24_to_tminus23', ..., 'e0.5_slope_tminus1_to_t', 'p_tminus24', ..., 'p_t', 'p_slope_tminus24_to_tminus23', ..., 'p_slope_tminus1_to_t']
    """
    feature_names = []  # Initialize the list to store generated feature names

    # Loop through each input type specified in inputs_to_use
    for input_type in inputs_to_use:
        # Generate feature names for lagged inputs from tminus24 to t
        feature_names += [f'{input_type}_tminus{i}' for i in range(24, 0, -1)] + [f'{input_type}_t']

    # If add_slope is True, add slope feature names for the current input type
    if add_slope:
        for input_type in inputs_to_use:
            # Add slope names from tminus24 to tminus1
            feature_names += [f'{input_type}_slope_tminus{i}_to_tminus{i - 1}' for i in range(24, 1, -1)]
            # Add slope name from tminus1 to t
            feature_names.append(f'{input_type}_slope_tminus1_to_t')

    # Return the list of generated feature names
    return feature_names


def generate_slope_column_names(inputs_to_use: List[str], padding: bool = False) -> List[str]:
    """
    Generate slope column names for each input type based on the specified time steps, optionally including padding.

    For each input type, this function generates column names for slopes calculated between
    consecutive time steps. It includes slopes from 'tminus24' to 'tminus2', a special case
    for the slope from 'tminus1' to 't', and optionally a padding slope at the beginning.

    Parameters:
    - inputs_to_use (List[str]): A list of input types for which to generate slope column names.
    - padding (bool): Whether to include an additional column name for padding at the beginning.

    Returns:
    - List[str]: A list of slope column names for all specified input types, including padding if specified.
    """

    slope_column_names = []
    for input_type in inputs_to_use:
        # Optionally add padding slope column name
        if padding:
            slope_column_names.append(f'{input_type}_slope_padding_to_tminus24')
        # Slopes from tminus24 to tminus2
        for i in range(24, 1, -1):
            slope_column_names.append(f'{input_type}_slope_tminus{i}_to_tminus{i - 1}')
        # Slope from tminus1 to t
        slope_column_names.append(f'{input_type}_slope_tminus1_to_t')

    return slope_column_names


def compute_slope(data: pd.DataFrame, input_type: str, padding: bool = False) -> np.ndarray:
    """
    Compute the slope for a given input type across its time-series columns and optionally
    pad the slope series to match the original time series length by duplicating the first slope.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the time series data.
    - input_type (str): The input type for which to compute the slope (e.g., 'e0.5').
    - padding (bool, optional): If True, duplicate the first slope to match the original time series length.
      Defaults to False.

    Returns:
    - np.ndarray: Array of slopes for the specified input type. If padding is True, the length of this
      array matches the length of the original time series.
    """
    # Define the column names based on the input_type
    columns = [f'{input_type}_tminus{i}' for i in range(24, 0, -1)] + [f'{input_type}_t']
    input_values = data[columns].values

    # Compute the slope between consecutive columns
    slopes = np.diff(input_values, axis=1)

    # print shape of slopes
    # print(slopes.shape)

    if padding:
        # Duplicate the first slope value if padding is True
        first_slope = slopes[:, 0].reshape(-1, 1)
        slopes = np.hstack([first_slope, slopes])

        # print shape of slopes
        # print(slopes.shape)

    return slopes


def check_nan_in_dataset(dataset: np.ndarray, dataset_name: str) -> None:
    """
    Checks if there are any NaN values in the given dataset and prints the result.

    Parameters:
    - dataset (np.ndarray): The dataset to check for NaN values.
    - dataset_name (str): The name of the dataset (for display purposes).
    """
    if np.isnan(dataset).any():
        print(f"NaN values found in {dataset_name}.")
    else:
        print(f"No NaN values in {dataset_name}.")


def min_max_norm(data: pd.DataFrame or pd.Series) -> pd.DataFrame or pd.Series:
    """
    Apply min-max normalization to a pandas DataFrame or Series.
    If the min and max values of a column are the same, that column is replaced with zeros.

    Parameters:
    - cme_files (pd.DataFrame or pd.Series): The pandas DataFrame or Series to be normalized.

    Returns:
    - pd.DataFrame or pd.Series: Min-max normalized pandas DataFrame or Series.
    """

    # Function to normalize a single column
    def normalize_column(column: pd.Series) -> pd.Series:
        min_val = column.min()
        max_val = column.max()

        # Handle case where max and min are the same
        if min_val == max_val:
            return pd.Series(np.zeros_like(column), index=column.index)
        else:
            # Apply min-max normalization
            return (column - min_val) / (max_val - min_val)

    # Check if the input is a DataFrame
    if isinstance(data, pd.DataFrame):
        normalized_df = data.apply(normalize_column, axis=0)
        return normalized_df

    # Check if the input is a Series
    elif isinstance(data, pd.Series):
        return normalize_column(data)

    else:
        raise TypeError("Input must be a pandas DataFrame or Series")


def preprocess_cme_features(df: pd.DataFrame, inputs_to_use: List[str]) -> pd.DataFrame:
    """
    Apply efficient preprocessing steps to the given dataframe based on the specified scheme table and inputs_to_use.

    Parameters:
    - df (pd.DataFrame): The dataframe to preprocess.
    - inputs_to_use (List[str]): List of input types to include in the dataset.

    Returns:
    - pd.DataFrame: The preprocessed dataframe.
    """

    # Preallocate a dictionary to store preprocessed cme_files
    preprocessed_data = {}

    # Define a mapping for intensity columns based on inputs_to_use
    intensity_mapping = {
        'e0.5': 'e0.5_max_intensity',
        'e1.8': 'e1.8_max_intensity',
        'p': 'p_max_intensity',
        'e4.4': 'e4.4_max_intensity',
        'p6.1': 'p6.1_max_intensity',
        'p33.0': 'p33.0_max_intensity'
    }

    # Natural Log Transformations for selected intensity columns
    for input_type in inputs_to_use:
        intensity_column = intensity_mapping.get(input_type)
        if intensity_column:
            log_intensity_column = f'log_{intensity_column}'
            preprocessed_data[f'log_{intensity_column}'] = np.log1p(df[intensity_column])
            # Apply Min-Max normalization on log-transformed features
            preprocessed_data[f'{log_intensity_column}_norm'] = min_max_norm(preprocessed_data[log_intensity_column])
            # Drop the original log-transformed column as it's not needed after normalization
            preprocessed_data.pop(log_intensity_column)

    preprocessed_data['log_half_richardson_value'] = np.log1p(-df['half_richardson_value'])
    preprocessed_data['log_diffusive_shock'] = np.log1p(df['diffusive_shock'])
    preprocessed_data['log_Type2_Viz_Area'] = df['Type2_Viz_Area'].apply(lambda x: np.log(x) if x != 0 else np.log(1))

    # Apply Min-Max normalization on all features, including the log-transformed ones
    for feature, proper_name in {'VlogV': 'VlogV', 'CME_DONKI_speed': 'CME_DONKI_speed',
                                 'Linear_Speed': 'CME_CDAW_LinearSpeed',
                                 '2nd_order_speed_final': '2nd_order_speed_final',
                                 '2nd_order_speed_20R': '2nd_order_speed_20R',
                                 'CMEs_with_speed_over_1000_in_past_9hours': 'CMEs Speed > 1000',
                                 'max_CME_speed_in_past_day': 'Max CME Speed', 'CMEs_in_past_month': 'CMEs Past Month',
                                 'CME_DONKI_longitude': 'CME_DONKI_longitude', 'CME_CDAW_MPA': 'CME_CDAW_MPA',
                                 'daily_sunspots': 'Sunspot Number', 'DONKI_half_width': 'DONKI_half_width',
                                 'CME_DONKI_latitude': 'CME_DONKI_latitude', 'Accelaration': 'Accelaration',
                                 'CPA': 'CPA', 'CMEs_in_past_9hours': 'CMEs Past 9 Hours'}.items():
        preprocessed_data[f"{feature}_norm"] = min_max_norm(df[proper_name])

    preprocessed_data['log_richardson_value_norm'] = min_max_norm(preprocessed_data['log_half_richardson_value'])
    preprocessed_data['log_diffusive_shock_norm'] = min_max_norm(preprocessed_data['log_diffusive_shock'])
    preprocessed_data['log_Type2_Viz_Area_norm'] = min_max_norm(preprocessed_data['log_Type2_Viz_Area'])

    # No transformation for 'Halo'
    preprocessed_data['Halo'] = df['Halo']

    # drop log_richardson_formula_1.0_c, diffusive shock, log_Type_2_Area because they are not needed anymore
    preprocessed_data.pop('log_half_richardson_value')
    preprocessed_data.pop('log_diffusive_shock')
    preprocessed_data.pop('log_Type2_Viz_Area')

    return pd.DataFrame(preprocessed_data)


def zero_out_cme_below_threshold(df: pd.DataFrame, threshold: float, cme_columns: List[str]) -> pd.DataFrame:
    """
    Zeroes out the values of specified CME columns in rows where the CME speed is below the threshold.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the cme_files.
    - threshold (float): The CME speed threshold.
    - cme_columns (List[str]): List of CME column names to zero out.

    Returns:
    - pd.DataFrame: The DataFrame with updated CME columns.
    """
    mask = df['CME_DONKI_speed'] < threshold
    for column in cme_columns:
        df.loc[mask, column] = 0
    return df


def create_synthetic_dataset(n_samples: int = 50000, n_timesteps: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a synthetic dataset consisting of sequences with a flat line segment followed by an angled line segment.
    Each sequence can have varying angles for the angled line segment, and noise is added for realism.

    Parameters:
    n_samples (int): The number of samples to generate in the dataset.
    n_timesteps (int): The total number of timesteps in each sample.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the normalized synthetic dataset and the corresponding targets.
    """
    synthetic_data = np.zeros((n_samples, n_timesteps, 1))
    targets = np.zeros((n_samples, 1))

    for i in range(n_samples):
        # Randomly determine the length of the flat and angled segments
        flat_length = np.random.randint(1, n_timesteps - 1)
        angle_length = n_timesteps - flat_length

        # Create a flat line segment
        flat_segment = np.zeros((flat_length, 1))

        # Randomly choose an angle between 30 and 60 degrees for the angled segment
        angle = np.random.randint(30, 61)
        radians = np.deg2rad(angle)
        tan = np.tan(radians)
        angle_segment = tan * np.arange(0, angle_length).reshape(-1, 1)

        # Add random noise to the angled segment
        noise = np.random.normal(0, 0.1, size=angle_segment.shape)
        angle_segment += noise

        # Concatenate the flat and angled segments
        full_segment = np.concatenate((flat_segment, angle_segment), axis=0)

        # Assign the concatenated segment and target
        synthetic_data[i, :, 0] = full_segment.ravel()
        targets[i] = angle

    # Normalize the synthetic cme_files
    synthetic_data = (synthetic_data - np.mean(synthetic_data)) / np.std(synthetic_data)

    # Normalize the targets
    targets = (targets - np.mean(targets)) / np.std(targets)

    return synthetic_data, targets


def plot_conv1d_filters(model: Model, layer_index: int, num_columns: int = 3):
    """
    Plots the filters of a 1D convolutional layer in a grid.

    Parameters:
    model (Model): The trained Keras model.
    layer_index (int): The index of the convolutional layer to visualize.
    num_columns (int): Number of columns in the grid for displaying filters. Default is 3.
    """
    # Extracting the filters and biases from the specified layer
    filters, biases = model.layers[layer_index].get_weights()

    # Number of filters
    num_filters = filters.shape[2]

    # Number of rows to display
    num_rows = num_filters // num_columns + (num_filters % num_columns > 0)

    # Setting up the plot
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 5, num_rows * 4))
    axes = axes.flatten()

    # Plotting each filter
    for i in range(num_filters):
        # Get the filter
        # print(f'filter shape: {filters.shape}')
        f = filters[:, :, i]  # .flatten()  # Flatten to 1D array if necessary

        # Plotting
        ax = axes[i]
        ax.plot(f, label=f'Weights in Filter {i}')
        ax.set_title(f'Filter {i}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Weight Value')

        # Show the legend for the first filter only, for clarity
        if i == 0:
            ax.legend()

    # Hide unused axes
    for ax in axes[num_filters:]:
        ax.axis('off')

    plt.tight_layout()
    plt.suptitle(f'Convolutional Layer {layer_index} Filters Visualization', fontsize=16, y=1.02)
    plt.show()


def plot_sample(data: np.ndarray, sample_index: int = None) -> None:
    """
    Plots a random sample from the synthetic dataset. If a sample index is provided,
    plots that specific sample.

    Parameters:
    cme_files (np.ndarray): The synthetic dataset with shape (n_samples, n_timesteps, 1).
    sample_index (int, optional): The index of the specific sample to plot. If None, a random
                                  sample will be plotted. Defaults to None.
    """

    if sample_index is None:
        sample_index = np.random.randint(low=0, high=data.shape[0])

    # Extract the specific sample
    sample = data[sample_index, :, 0]  # Remove the last dimension as it's 1

    # Plot the sample
    plt.figure(figsize=(10, 4))
    plt.plot(sample, marker='o')
    plt.title(f'Synthetic Data Sample at Index {sample_index}')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()


def exponential_formatter(x, pos):
    """Custom formatter to convert log values back to exponential values for labeling."""
    return f'{np.exp(x):.2f}'


def read_sep_data(file_path: str) -> pd.DataFrame:
    """
    Reads the SEP event cme_files from a CSV file.

    Parameters:
    - file_path (str): The path to the SEP event CSV file.

    Returns:
    - DataFrame: The SEP event cme_files as a pandas DataFrame.
    """
    return pd.read_csv(file_path)


def normalize_flux(df: pd.DataFrame, columns: List[str], apply_log: bool = True) -> pd.DataFrame:
    """
    Normalizes the specified flux intensity columns in the DataFrame between 0 and 1.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the SEP event cme_files.
    - columns (List[str]): A list of column names to be normalized.
    - apply_log (bool): Whether to apply a logarithmic transformation. Default is True.

    Returns:
    - DataFrame: The DataFrame with the normalized columns.
    """
    for column in columns:
        if apply_log:
            # Apply a logarithmic transformation before normalization
            df[column] = np.log1p(df[column])  # Adding 1 to avoid log(0)
        df[column] = min_max_norm(df[column])
    return df


def extract_cme_start_times(df: pd.DataFrame) -> List[pd.Timestamp]:
    """
    Extracts CME start times from the 'cme_donki_time' column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the SEP event cme_files.

    Returns:
    - List[pd.Timestamp]: A list of CME start times.
    """
    valid_times = []
    for time in df['cme_donki_time']:
        if time != 0:  # Exclude zero values which are placeholders
            try:
                valid_time = pd.to_datetime(time)
                valid_times.append(valid_time)
            except (ValueError, TypeError):
                # Skip invalid or non-datetime values
                # print(f"ValueError or TypeError for time: {time}")
                continue
    return valid_times


def plot_and_evaluate_sep_event(
        df: pd.DataFrame,
        cme_start_times: List[pd.Timestamp],
        event_id: str,
        model: tf.keras.Model,
        input_columns: List[str],
        using_cme: bool = False,
        title: str = None,
        inputs_to_use: List[str] = None,
        add_slope: bool = True,
        outputs_to_use: List[str] = None,
        show_persistent: bool = True,
        show_changes: bool = True,
        prefix: str = 'testing',
        use_dict: bool = False) -> Tuple[float, str]:
    """
    Plots the SEP event cme_files with actual and predicted proton intensities, electron intensity,
    and evaluates the model's performance using MAE.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the SEP event cme_files with normalized values.
    - cme_start_times (List[pd.Timestamp]): A list of CME start times for vertical markers.
    - event_id (str): The event ID to be displayed in the plot title.
    - model (tf.keras.Model): The trained model to be evaluated.
    - input_columns (List[str]): The list of input columns for the model.
    - cme_columns (List[str]): The list of CME feature columns.
    - target_column (str): The column name of the target variable.
    - using_cme (bool): Whether to use CME features. Default is False.
    - model_type (str): The type of model used. Default is 'cnn'.
    - title (str): The title of the plot. Default is None.
    - inputs_to_use (List[str]): The list of input types to use. Default is None.
    - add_slope (bool): Whether to add slope features. Default is True.
    - outputs_to_use (List[str]): The list of output types to use. Default is None.
    - show_persistent (bool): Whether to show the persistent model where the delta = 0. Default is True.
    - show_changes (bool): Whether to show the scatter plot of target changes vs predicted changes. Default is True.
    - prefix (bool): Whether to add the prefix 'SEP Event' to the title. Default is True.
    - use_dict (bool): Whether to use the dictionary for the model. Default is False.

    Returns:
    - Tuple[float, str]: A tuple containing the MAE loss and the plot title.
    """
    global actual_changes, predicted_changes, delta_count, selected_changes, mask, e44_intensity_log, p61_intensity_log, p330_intensity_log
    e18_intensity_log = None

    if add_slope:
        n_features = len(inputs_to_use) * (25 + 24)
    else:
        n_features = len(inputs_to_use) * 25

    # Extract and adjust the timestamp for plotting
    timestamps = pd.to_datetime(df['Timestamp']) + pd.Timedelta(minutes=30)
    t_timestamps = pd.to_datetime(df['Timestamp'])
    # Transform 't' column to log scale and plot
    e05_intensity_log = np.log1p(df['e0.5_t'])  # Using log1p for numerical stability
    p_t_log = np.log1p(df['p_t'])  # Using log1p for numerical stability

    if 'e1.8' in inputs_to_use:
        e18_intensity_log = np.log1p(df['e1.8_t'])  # Using log1p for numerical stability

    if 'e4.4' in inputs_to_use:
        e44_intensity_log = np.log1p(df['e4.4_t'])

    if 'p6.1' in inputs_to_use:
        p61_intensity_log = np.log1p(df['p6.1_t'])

    if 'p33.0' in inputs_to_use:
        p330_intensity_log = np.log1p(df['p33.0_t'])

    # Normalize the flux intensities
    df_norm = normalize_flux(df, input_columns, apply_log=True)
    # X = df_norm[input_columns].values
    added_columns = []
    if add_slope:
        added_columns = generate_slope_column_names(inputs_to_use)
        for input_type in inputs_to_use:
            slope_values = compute_slope(df_norm, input_type)
            slope_column_names = generate_slope_column_names([input_type])
            for slope_column, slope_index in zip(slope_column_names, range(slope_values.shape[1])):
                df_norm[slope_column] = slope_values[:, slope_index]

    X = df_norm[input_columns + added_columns].values

    target_columns = []
    if 'delta_p' in outputs_to_use:
        target_columns.append('delta_log_Intensity')
    if 'p' in outputs_to_use:
        target_columns.append('Proton Intensity')

    # log the intensity but not delta
    df['Proton Intensity'] = np.log1p(df['Proton Intensity'])
    # y_true = df[target_columns].values
    actual_proton = df['Proton Intensity'].values

    if using_cme:
        # process cme features
        cme_features = preprocess_cme_features(df, inputs_to_use)
        X_reshaped = np.concatenate([X, cme_features.values], axis=1)
    else:
        X_reshaped = X.reshape(-1, n_features, 1)

    if add_slope:
        # n_features_list = [25] * len(inputs_to_use) * 2
        n_features_list = [25] * len(inputs_to_use) + [24] * len(inputs_to_use)
    else:
        n_features_list = [25] * len(inputs_to_use)

    X_reshaped = reshape_X(X_reshaped, n_features_list, inputs_to_use, add_slope, 'mlp')

    # Evaluate the model
    if use_dict:
        res = model.predict(X_reshaped)
        predictions = res['output']
    else:
        _, predictions = model.predict(X_reshaped)
    predictions = process_predictions(predictions)

    # if target change then we need to convert prediction into actual value
    if 'p' in inputs_to_use and 'delta_p' in outputs_to_use:
        predictions_plot = p_t_log + predictions
        if show_changes:
            actual_changes = df['delta_log_Intensity'].values - 1  # offset by 1
            actual_changes_nooffset = df['delta_log_Intensity'].values
            mask = (actual_changes_nooffset >= -0.01) & (actual_changes_nooffset <= 0.01)
            selected_changes = actual_changes_nooffset[mask]
            predicted_changes = predictions - 1  # offset by 1
    else:
        predictions_plot = predictions

    threshold_value = 0.4535
    mae_loss = mean_absolute_error(actual_proton, predictions_plot)
    # t_lag, s_lag, avg_lag = evaluate_lag_error(timestamps, y_true, predictions_plot, threshold=threshold_value)
    # print(f"Threshold lag error: {t_lag:.2f} minutes")
    # print(f"Shift lag error: {s_lag:.2f} minutes")
    # print(f"Average lag error: {avg_lag:.2f} minutes")
    # mae_loss = mean_absolute_error(y_true, p_t_log) # simple model
    print(f"Mean Absolute Error (MAE) on the cme_files: {mae_loss}")

    lw = .65  # Line width for the plots
    tlw = 1.8  # Thicker line width for the actual and predicted lines
    ssz = 4.7  # Scatter size for the changes

    # Plot the cme_files
    plt.figure(figsize=(15, 10), facecolor='white')
    plt.plot(timestamps, actual_proton, label='Actual Proton ln(Intensity)', color='blue', linewidth=tlw)
    plt.plot(timestamps, predictions_plot, label='Predicted Proton ln(Intensity)', color='red', linewidth=tlw)
    plt.plot(t_timestamps, e05_intensity_log, label='E 0.5 ln(Intensity)', color='orange', linewidth=lw)
    if 'e1.8' in inputs_to_use:
        plt.plot(t_timestamps, e18_intensity_log, label='E 1.8 ln(Intensity)', color='yellow', linewidth=lw)
    if 'e4.4' in inputs_to_use:
        plt.plot(t_timestamps, e44_intensity_log, label='E 4.4 ln(Intensity)', color='teal', linewidth=lw)
    if 'p6.1' in inputs_to_use:
        plt.plot(t_timestamps, p61_intensity_log, label='P 6.1 ln(Intensity)', color='purple', linewidth=lw)
    if 'p33.0' in inputs_to_use:
        plt.plot(t_timestamps, p330_intensity_log, label='P 33.0 ln(Intensity)', color='brown', linewidth=lw)
    if 'p' in inputs_to_use and 'delta_p' in outputs_to_use and show_persistent:
        plt.plot(timestamps, p_t_log, label='Persistent Model', color='black', linestyle=':', linewidth=tlw)
    if 'p' in inputs_to_use and 'delta_p' in outputs_to_use and show_changes:
        plt.scatter(timestamps, actual_changes, color='gray', label='Actual Changes', alpha=0.5, s=ssz)
        plt.scatter(timestamps, predicted_changes, color='purple', label='Predicted Changes', alpha=0.5, s=ssz)
        # Add a black horizontal line at log(0.05) on the y-axis and create a handle for the legend

        # print count of changes before mask and after mask
        # print(f"Count of changes before mask: {len(actual_changes)}")
        # print(f"Count of changes after mask: {len(selected_changes)}")
        # print(f'Display masked: {actual_changes.flatten()[mask]}')

        # Plot the actual changes within the range, offset by -2
        plt.scatter(timestamps[mask], selected_changes - 2, color='green', label='delta in (-0.01, 0.01)',
                    alpha=0.5, s=ssz)

        # Count the number of actual changes within the range
        delta_count = np.sum(mask)
        # Extract handles and labels for the plot's elements
        handles, labels = plt.gca().get_legend_handles_labels()
        labels.extend(f' (Count: {delta_count})')

    plt.axhline(y=threshold_value, color='black', linestyle='--', linewidth=lw, label='Threshold')

    # Create a custom legend handle for the CME start times
    cme_line = Line2D([0], [0], color='green', linestyle='--', linewidth=lw, label='CME Start Time')

    # Add vertical markers for CME start times, also adjusted by 30 minutes
    for cme_time in cme_start_times:
        if pd.notna(cme_time):  # Check if cme_time is a valid timestamp
            plt.axvline(x=cme_time, color='green', linestyle='--', linewidth=lw)

    # Set custom tick labels for y-axis to represent exponential values
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(exponential_formatter))

    plt.xlabel('Timestamp')
    plt.ylabel('Ln Flux lnIntensity')
    # plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.title(f'{title} - SEP Event {event_id} - MAE: {mae_loss:.4f}')  # - Lag: {s_lag:.2f} minutes')

    # Extract handles and labels for the plot's elements
    handles, labels = plt.gca().get_legend_handles_labels()
    # Add custom legend handles for the threshold and CME lines
    handles.extend([cme_line])
    labels.extend(["CME Start Time"])

    plt.legend(handles=handles, labels=labels)

    # Save the plot to a file with the MAE in the file name
    file_name = f'{title}_SEP_Event_{event_id}_{prefix}_MAE_{mae_loss:.4f}.png'  # _Lag_{s_lag:.2f}.png'
    plt.savefig(file_name)

    # plt.show()
    # Close the plot
    plt.close()

    # Return the file location
    file_location = os.path.abspath(file_name)
    print(f"Saved plot to: {file_location}")
    return mae_loss, file_name


def plot_avsp_delta(
        df: pd.DataFrame,
        model: tf.keras.Model,
        input_columns: List[str],
        using_cme: bool = False,
        inputs_to_use: List[str] = None,
        add_slope: bool = True,
        outputs_to_use: List[str] = None,
        use_dict: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes actual vs predicted delta values and ln intensities.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the SEP event data with normalized values.
    - model (tf.keras.Model): The trained model to be evaluated.
    - input_columns (List[str]): The list of input columns for the model.
    - using_cme (bool): Whether to use CME features. Default is False.
    - inputs_to_use (List[str]): The list of input types to use. Default is None.
    - add_slope (bool): Whether to add slope features. Default is True.
    - outputs_to_use (List[str]): The list of output types to use. Default is None.
    - use_dict (bool): Whether to use the dictionary for the model. Default is False.

    Returns:
    - Tuple containing:
        - actual_changes (np.ndarray): Actual delta values.
        - predicted_changes (np.ndarray): Predicted delta values.
        - actual_ln_intensity (np.ndarray): Actual ln intensity values.
        - predicted_ln_intensity (np.ndarray): Predicted ln intensity values.
    """

    if add_slope:
        n_features = len(inputs_to_use) * (25 + 24)
    else:
        n_features = len(inputs_to_use) * 25

    p_t_log = np.log1p(df['p_t'])  # Using log1p for numerical stability

    # Normalize the flux intensities
    df_norm = normalize_flux(df, input_columns, apply_log=True)

    # Add slope columns if required
    added_columns = []
    if add_slope:
        added_columns = generate_slope_column_names(inputs_to_use)
        for input_type in inputs_to_use:
            slope_values = compute_slope(df_norm, input_type)
            slope_column_names = generate_slope_column_names([input_type])
            for slope_column, slope_index in zip(slope_column_names, range(slope_values.shape[1])):
                df_norm[slope_column] = slope_values[:, slope_index]

    # Prepare input data
    X = df_norm[input_columns + added_columns].values

    # Prepare target columns
    target_columns = []
    if 'delta_p' in outputs_to_use:
        target_columns.append('delta_log_Intensity')
    if 'p' in outputs_to_use:
        target_columns.append('Proton Intensity')

    # Apply ln transformation to 'Proton Intensity'
    actual_ln_intensity = np.log1p(df['Proton Intensity'])

    # Handle CME features if used
    if using_cme:
        cme_features = preprocess_cme_features(df, inputs_to_use)
        X_reshaped = np.concatenate([X, cme_features.values], axis=1)
    else:
        X_reshaped = X.reshape(-1, n_features, 1)

    # Define feature lengths for reshaping
    if add_slope:
        n_features_list = [25] * len(inputs_to_use) + [24] * len(inputs_to_use)
    else:
        n_features_list = [25] * len(inputs_to_use)

    # Reshape input data
    X_reshaped = reshape_X(X_reshaped, n_features_list, inputs_to_use, add_slope, 'mlp')

    # Make predictions using the model
    if use_dict:
        res = model.predict(X_reshaped)
        predictions = res['output']
    else:
        _, predictions = model.predict(X_reshaped)

    predictions = process_predictions(predictions)

    # Extract actual delta values
    actual_changes = df['delta_log_Intensity'].values
    predicted_changes = predictions

    # Compute predicted ln intensity by adding predicted delta to p_t_log
    predicted_ln_intensity = p_t_log + predicted_changes

    # Print shapes and first 5 elements of each array
    # print("\nArray shapes:")
    # print(f"actual_changes shape: {actual_changes.shape}")
    # print(f"predicted_changes shape: {predicted_changes.shape}")
    # print(f"actual_ln_intensity shape: {actual_ln_intensity.shape}")
    # print(f"predicted_ln_intensity shape: {predicted_ln_intensity.shape}")

    # print("\nFirst 5 elements of each array:")
    # print("Row | actual_changes | predicted_changes | actual_ln_int | predicted_ln_int")
    # print("-" * 65)
    # for i in range(5):
    #     print(f"{i:3d} | {actual_changes[i]:13.6f} | {predicted_changes[i]:16.6f} | "
    #           f"{actual_ln_intensity[i]:11.6f} | {predicted_ln_intensity[i]:14.6f}")

    # print("\nValue ranges:")
    # print(f"actual_changes: min={actual_changes.min():.6f}, max={actual_changes.max():.6f}")
    # print(f"predicted_changes: min={predicted_changes.min():.6f}, max={predicted_changes.max():.6f}")
    # print(f"actual_ln_intensity: min={actual_ln_intensity.min():.6f}, max={actual_ln_intensity.max():.6f}")
    # print(f"predicted_ln_intensity: min={predicted_ln_intensity.min():.6f}, max={predicted_ln_intensity.max():.6f}")

    return actual_changes, predicted_changes, actual_ln_intensity, predicted_ln_intensity


def process_sep_events(
        directory: str,
        model: tf.keras.Model,
        using_cme: bool = False,
        title: str = None,
        inputs_to_use: List[str] = None,
        add_slope: bool = True,
        cme_speed_threshold: float = 0,
        outputs_to_use: List[str] = None,
        show_avsp: bool = False,
        show_error_hist: bool = True,
        show_error_concentration: bool = True,
        prefix: str = 'testing',
        use_dict: bool = False) -> List[str]:
    """
    Processes SEP event files in the specified directory, normalizes flux intensities, predicts proton intensities,
    plots the results, and calculates the MAE for each file.

    Parameters:
    - directory (str): Path to the directory containing the SEP event files.
    - model (tf.keras.Model): The trained machine learning model for predicting proton intensity.
    - using_cme (bool): Whether to use CME features. Default is False.
    - title (str): The title of the plot. Default is None.
    - inputs_to_use (List[str]): List of input types to include in the dataset.
    - add_slope (bool): If True, adds slope features to the dataset.
    - outputs_to_use (List[str]): List of output types to include in the dataset.
    - cme_speed_threshold (float): The threshold for CME speed. CMEs with speeds below this threshold will be excluded.
    - show_avsp (bool): Whether to show the Actual vs Predicted delta plot. Default is False.
    - show_error_hist (bool): Whether to show the error histogram. Default is True.
    - prefix (str): The prefix to use for the plot file names. Default is 'testing'.
    - use_dict (bool): Whether to use the dictionary for the model. Default is False.

    Returns:
    - List[str]: A list containing the names of the plot files.

    The function assumes that the SEP event files are named in the format 'sep_event_X_filled_ie.csv',
    where 'X' is the event ID. It skips files where the proton intensity is -9999.
    Each file will be processed to plot actual vs predicted proton intensity and electron intensity.
    A MAE score will be printed for each file.
    """
    if inputs_to_use is None:
        inputs_to_use = ['e0.5', 'e4.4', 'p6.1', 'p']

    if outputs_to_use is None:
        outputs_to_use = ['delta_p', 'p']

    # Generate input columns based on inputs_to_use and add slope columns if add_slope is True
    input_columns = []
    for input_type in inputs_to_use:
        input_columns += [f'{input_type}_tminus{i}' for i in range(24, 0, -1)] + [f'{input_type}_t']

    target_column = 'Proton Intensity'

    plot_names = []

    cme_columns_to_zero_out = [
        'CME_DONKI_latitude', 'CME_DONKI_longitude', 'CME_DONKI_speed', 'CME_CDAW_MPA',
        'CME_CDAW_LinearSpeed', 'VlogV', 'DONKI_half_width', 'Accelaration',
        '2nd_order_speed_final', '2nd_order_speed_20R', 'CPA', 'Halo', 'Type2_Viz_Area',
        'solar_wind_speed', 'diffusive_shock', 'half_richardson_value'
    ]

    # Initialize lists to hold data for plotting
    avsp_data_delta = []
    avsp_data_intensity = []

    # Iterate over files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('_ie_trim.csv'):
            try:
                file_path = os.path.join(directory, file_name)

                # Read the SEP event data
                df = read_sep_data(file_path)

                # Skip files where proton intensity is -9999
                if (df[target_column] == -9999).any():
                    continue

                df = zero_out_cme_below_threshold(df, cme_speed_threshold, cme_columns_to_zero_out)

                # Extract CME start times
                cme_start_times = extract_cme_start_times(df)
                # Extract event ID from filename
                event_id = file_name.split('_')[2]

                # save a copy of the dataframe to use for showing the actual vs predicted plot
                df_for_avsp = df.copy()

                # Plot and evaluate the SEP event
                mae_loss, plotname = plot_and_evaluate_sep_event(
                    df, cme_start_times, event_id, model,
                    input_columns, using_cme=using_cme,
                    title=title, inputs_to_use=inputs_to_use,
                    add_slope=add_slope, outputs_to_use=outputs_to_use,
                    prefix=prefix, use_dict=use_dict)

                print(f"Processed file: {file_name} with MAE: {mae_loss}")
                plot_names.append(plotname)

                if show_avsp:
                    actual_ch, predicted_ch, actual_ln_intensity, predicted_ln_intensity = plot_avsp_delta(
                        df_for_avsp, model, input_columns, using_cme=using_cme,
                        inputs_to_use=inputs_to_use, add_slope=add_slope,
                        outputs_to_use=outputs_to_use, use_dict=use_dict)

                    # Collect data for delta plot
                    avsp_data_delta.append((event_id, actual_ch, predicted_ch))
                    # Collect data for ln intensity plot including actual_delta
                    avsp_data_intensity.append((event_id, actual_ln_intensity, predicted_ln_intensity, actual_ch))

            except Exception as e:
                print(f"Error processing file: {file_name}")
                print(e)
                traceback.print_exc()
                continue

    if show_avsp:
        # Plot delta-based actual vs predicted
        if avsp_data_delta:
            plot_file_location_delta = plot_actual_vs_predicted(
                avsp_data_delta, title + ' Delta', prefix + '_delta')
            print(f"Saved delta plot to: {plot_file_location_delta}")
            plot_names.append(plot_file_location_delta)

            # NOTE: not necessary for now
            # if show_error_hist:
            #     histogram_path = plot_error_dist(
            #         avsp_data_delta, f"{title} Delta Error Distribution", prefix + '_delta')
            #     print(f"Saved delta histogram plot to: {histogram_path}")
            #     plot_names.append(histogram_path)

            # if show_error_concentration:
            #     concentration_path = plot_error_concentration(
            #         avsp_data_delta, f"{title} Delta Error Concentration", prefix + '_delta')
            #     print(f"Saved delta concentration plot to: {concentration_path}")
            #     plot_names.append(concentration_path)

        # Plot ln intensity-based actual vs predicted
        if avsp_data_intensity:
            plot_file_location_intensity = plot_actual_vs_predicted_intensity(
                avsp_data_intensity, title + ' Ln Intensity', prefix + '_ln_intensity')
            print(f"Saved ln intensity plot to: {plot_file_location_intensity}")
            plot_names.append(plot_file_location_intensity)

            # NOTE: does not work for now
            # if show_error_hist:
            #     histogram_path = plot_error_dist(
            #         avsp_data_intensity, f"{title} Ln Intensity Error Distribution", prefix + '_ln_intensity')
            #     print(f"Saved ln intensity histogram plot to: {histogram_path}")
            #     plot_names.append(histogram_path)

            # if show_error_concentration:
            #     concentration_path = plot_error_concentration(
            #         avsp_data_intensity, f"{title} Ln Intensity Error Concentration", prefix + '_ln_intensity')
            #     print(f"Saved ln intensity concentration plot to: {concentration_path}")
            #     plot_names.append(concentration_path)

    return plot_names


def plot_error_dist(avsp_data: List[Tuple[str, np.ndarray, np.ndarray]], title: str, prefix: str) -> str:
    """
    Plots a histogram of error values derived from actual vs. predicted data pairs.

    Parameters:
    - avsp_data (List[Tuple[str, np.ndarray, np.ndarray]]): List of tuples containing event_id, actual data, and predicted data.
    - title (str): The title for the histogram plot.
    - prefix (str): Prefix for naming the plot file.

    Returns:
    - str: File path of the saved plot.
    """
    # Extract all errors from the avsp_data
    errors = []
    for _, actual, predicted in avsp_data:
        errors.extend(actual - predicted)  # Compute error as actual - predicted

    # Define the bin width and calculate the range of bins
    bin_width = 0.02
    bin_range = np.arange(min(errors), max(errors) + bin_width, bin_width)

    # Create and display the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=bin_range, color='blue', edgecolor='black', alpha=0.7)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True)

    plot_filename = f"{prefix}_error_distribution.png"
    plt.savefig(plot_filename)
    plt.close()

    return os.path.abspath(plot_filename)


def plot_error_concentration(avsp_data: List[Tuple[str, np.ndarray, np.ndarray]], title: str, prefix: str) -> str:
    """
    Plots a heatmap showing the concentration of errors across label bins.

    Parameters:
    - avsp_data (List[Tuple[str, np.ndarray, np.ndarray]]): List of tuples containing event_id, actual data, and predicted data.
    - title (str): The title for the heatmap plot.
    - prefix (str): Prefix for naming the plot file.

    Returns:
    - str: File path of the saved plot.
    """
    # Extract all errors and labels
    errors = []
    labels = []
    for _, actual, predicted in avsp_data:
        error = predicted - actual  # Compute error as actual - predicted
        errors.extend(error)
        labels.extend(actual)

    # Define the bin widths and ranges for errors and labels
    error_bin_width = 0.1
    label_bin_width = 0.4
    error_bins = np.arange(min(errors), max(errors) + error_bin_width, error_bin_width)
    label_bins = np.arange(min(labels), max(labels) + label_bin_width, label_bin_width)

    # Create a 2D histogram of errors and labels
    counts, xedges, yedges = np.histogram2d(errors, labels, bins=[error_bins, label_bins])

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    heatmap = plt.pcolormesh(xedges, yedges, counts.T, cmap='Greys')
    plt.colorbar(heatmap, label='Frequency')

    # Adding frequency labels to each cell
    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            plt.text(xedges[i] + error_bin_width / 2, yedges[j] + label_bin_width / 2, f'{int(counts[i][j])}',
                     color='tab:blue', ha='center', va='center', fontsize=8)

    plt.xlabel('Error')
    plt.ylabel('Label')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)

    plot_filename = f"{prefix}_error_concentration.png"
    plt.savefig(plot_filename)
    plt.close()

    return os.path.abspath(plot_filename)


def plot_actual_vs_predicted(avsp_data: List[tuple], title: str, prefix: str):
    """
    Plots actual vs predicted values for SEP events.

    Parameters:
    - avsp_data (List[tuple]): List of tuples containing event_id, actual data, and predicted data.
    - title (str): The title of the plot.
    - prefix (str): Prefix for the plot file names.
    """
    plt.figure(figsize=(10, 7))  # Adjust size as needed
    norm = plt.Normalize(-2.5, 2.5)
    cmap = plt.cm.viridis

    for event_id, actual, predicted in avsp_data:
        plt.scatter(actual, predicted, c=actual, cmap=cmap, norm=norm, label=f'{event_id}', alpha=0.7, s=12)

    plt.plot([-2.5, 2.5], [-2.5, 2.5], 'k--', label='Perfect Prediction')
    plt.xlabel('Actual Changes')
    plt.ylabel('Predicted Changes')
    plt.title(f"{title}\n{prefix}_Actual_vs_Predicted_Changes")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Actual Value', extend='both')
    plt.grid(True)

    plot_filename = f"{title}_{prefix}_actual_vs_predicted_changes.png"
    plt.savefig(plot_filename)
    plt.close()

    return os.path.abspath(plot_filename)


def plot_actual_vs_predicted_intensity(avsp_data: List[tuple], title: str, prefix: str, delta_threshold: float = 0.5):
    """
    Plots actual vs predicted intensity values for SEP events.

    Parameters:
    - avsp_data (List[tuple]): List of tuples containing event_id, actual ln(Intensity),
                               predicted ln(Intensity), and actual delta. The actual ln(Intensity),
                               predicted ln(Intensity), and actual delta can be arrays.
    - title (str): The title of the plot.
    - prefix (str): Prefix for the plot file names.
    - delta_threshold (float): Threshold for actual_delta to plot special markers. Default is 0.5.
    """

    plt.figure(figsize=(10, 7))  # Adjust size as needed

    # Initialize empty lists to hold flattened data
    all_actual_ln_intensities = []
    all_predicted_ln_intensities = []
    all_actual_deltas = []
    all_event_ids = []

    # Iterate over the data and flatten arrays
    for event_id, actual_ln_intensity, predicted_ln_intensity, actual_delta in avsp_data:
        # Flatten the arrays/lists if necessary
        actual_ln_intensity = np.array(actual_ln_intensity).flatten()
        predicted_ln_intensity = np.array(predicted_ln_intensity).flatten()
        actual_delta = np.array(actual_delta).flatten()

        # Append data to the overall lists
        all_actual_ln_intensities.extend(actual_ln_intensity)
        all_predicted_ln_intensities.extend(predicted_ln_intensity)
        all_actual_deltas.extend(actual_delta)
        all_event_ids.extend([event_id] * len(actual_ln_intensity))

    # Convert lists to NumPy arrays
    actual_ln_intensities = np.array(all_actual_ln_intensities)
    predicted_ln_intensities = np.array(all_predicted_ln_intensities)
    actual_deltas = np.array(all_actual_deltas)
    event_ids = np.array(all_event_ids)

    # Create a mask for points with actual_delta > delta_threshold
    mask_high_delta = actual_deltas > delta_threshold
    mask_low_delta = ~mask_high_delta

    # Set up color mapping based on actual_delta
    norm = plt.Normalize(np.min(actual_deltas), np.max(actual_deltas))
    cmap = plt.cm.viridis

    # Plot points with actual_delta <= delta_threshold
    plt.scatter(actual_ln_intensities[mask_low_delta], predicted_ln_intensities[mask_low_delta],
                c=actual_deltas[mask_low_delta], cmap=cmap, norm=norm,
                alpha=0.7, s=12)

    # Plot points with actual_delta > delta_threshold as "+" markers, plotted last
    plt.scatter(actual_ln_intensities[mask_high_delta], predicted_ln_intensities[mask_high_delta],
                c=actual_deltas[mask_high_delta], cmap=cmap, norm=norm,
                alpha=0.9, s=90, marker='+', linewidths=1.5)

    # Plot perfect prediction line
    min_intensity = min(np.min(actual_ln_intensities), np.min(predicted_ln_intensities))
    max_intensity = max(np.max(actual_ln_intensities), np.max(predicted_ln_intensities))
    plt.plot([min_intensity, max_intensity], [min_intensity, max_intensity], 'k--', label='Perfect Prediction')

    # Add dashed lines at ln(10) on both axes
    ln_10 = np.log(10)
    plt.axvline(ln_10, color='red', linestyle='--', label='SEP Threshold')
    plt.axhline(ln_10, color='red', linestyle='--')

    plt.xlabel('Actual ln(Intensity)')
    plt.ylabel('Predicted ln(Intensity)')
    plt.title(f"{title}\n{prefix}_Actual_vs_Predicted_Intensity")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, label='Actual Delta', extend='both')
    plt.grid(True)

    plt.legend()

    plot_filename = f"{title}_{prefix}_actual_vs_predicted_intensity.png"
    plt.savefig(plot_filename)
    plt.close()

    return os.path.abspath(plot_filename)


def plot_sample_with_cme(data: np.ndarray, cme_features_names: list = None, sample_index: int = None) -> None:
    """
    Plots a random sample from the dataset, showing both time-series and CME features.
    If a sample index is provided, plots that specific sample.

    Parameters:
    - cme_files (np.ndarray): The dataset with shape (n_samples, n_features).
    - cme_features_names (list): A list of names for the CME features.
    - sample_index (int, optional): The index of the specific sample to plot.
                                   If None, a random sample will be plotted. Defaults to None.
    """

    if sample_index is None:
        sample_index = np.random.randint(low=0, high=data.shape[0])

    if cme_features_names is None:
        cme_features_names = [
            'VlogV_norm', 'CME_DONKI_speed_norm', 'Linear_Speed_norm', '2nd_order_speed_final_norm',
            '2nd_order_speed_20R_norm', 'CMEs_with_speed_over_1000_in_past_9hours_norm',
            'max_CME_speed_in_past_day_norm', 'CMEs_in_past_month_norm', 'CME_DONKI_longitude_norm',
            'CME_CDAW_MPA_norm', 'daily_sunspots_norm', 'DONKI_half_width_norm',
            'CME_DONKI_latitude_norm', 'Accelaration_norm', 'CPA_norm', 'CMEs_in_past_9hours_norm',
            'log_e0.5_max_intensity_norm', 'log_e1.8_max_intensity_norm', 'log_p_max_intensity_norm',
            'log_richardson_value_norm', 'log_diffusive_shock_norm', 'log_Type2_Viz_Area_norm',
            'Halo']

    # Extract the specific sample
    sample = data[sample_index]

    # Split the sample into time-series and CME features
    time_series_data = sample[:75]  # Assuming first 25 features are time-series
    cme_features_count = len(cme_features_names)
    cme_features_data = sample[25:25 + cme_features_count]

    # Plot the time-series cme_files
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_series_data, marker='o')
    plt.title(f'Sample at Index {sample_index} - Time Series Data')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.grid(True)

    # Plot the CME features
    plt.subplot(2, 1, 2)
    plt.bar(range(cme_features_count), cme_features_data)
    plt.title('CME Features')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.xticks(range(cme_features_count), cme_features_names, rotation=90)  # Set custom x-axis ticks
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_sample_with_cme(data: np.ndarray, cme_start_index: int, cme_features_names: list = None,
                         sample_index: int = None) -> None:
    """
    Plots a random sample from the dataset, showing both time-series and CME features.
    If a sample index is provided, plots that specific sample.

    Parameters:
    - cme_files (np.ndarray): The dataset with shape (n_samples, n_features).
    - cme_start_index (int): The index where CME features start in the cme_files.
    - cme_features_names (list): A list of names for the CME features.
    - sample_index (int, optional): The index of the specific sample to plot.
                                   If None, a random sample will be plotted. Defaults to None.
    """

    if sample_index is None:
        sample_index = np.random.randint(low=0, high=data.shape[0])

    if cme_features_names is None:
        cme_features_names = ['CME Feature ' + str(i) for i in range(cme_start_index, data.shape[1])]

    # Extract the specific sample
    sample = data[sample_index]

    # Split the sample into time-series and CME features
    time_series_data = sample[:cme_start_index]
    cme_features_data = sample[cme_start_index:]

    # Plot the time-series cme_files
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_series_data, marker='o')
    plt.title(f'Sample at Index {sample_index} - Time Series Data')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.grid(True)

    # Plot the CME features
    plt.subplot(2, 1, 2)
    plt.bar(range(len(cme_features_data)), cme_features_data)
    plt.title('CME Features')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.xticks(range(len(cme_features_data)), cme_features_names, rotation=90)  # Set custom x-axis ticks
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def process_predictions(predictions: np.ndarray) -> np.ndarray:
    """
    Processes model predictions to ensure compatibility with models that output either single or multiple channels.
    In case of multiple channels, it extracts the predictions corresponding to the first channel.

    Parameters:
    - predictions (np.ndarray): The predictions made by the model. Can be either 1D or 2D (for models with multiple outputs).
    Returns:
    -
    """
    # Handle predictions based on their shape
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        # More than one channel: select the first channel
        processed_predictions = predictions[:, 0].flatten()
    else:
        # Single channel: just flatten it
        processed_predictions = predictions.flatten()

    return processed_predictions


def plot_error_hist(
        model: Model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sample_weights: Optional[np.ndarray],
        title: str,
        prefix: str) -> str:
    """
    Plots a histogram of the sum of prediction errors for a test dataset using a trained model.
    Optionally weights these sums if sample weights are provided.

    :param model: The trained model to evaluate.
    :param X_test: The test features.
    :param y_test: The true target values for the test set.
    :param sample_weights: The sample weights for the test set.
    :param title: The title of the plot.
    :param prefix: The prefix for the plot file name.

    :return: The file path of the saved plot.
    """
    # Predict the outputs
    _, predictions = model.predict(X_test)
    # Process predictions
    predictions = process_predictions(predictions)
    y_test = process_predictions(y_test)
    # Calculate squared errors
    squared_errors = (predictions - y_test) ** 2
    # Calculate weighted squared errors if sample weights are provided
    if sample_weights is not None:
        weighted_squared_errors = squared_errors * sample_weights
    else:
        weighted_squared_errors = squared_errors

    # Binning by label values with an interval width of 0.2, adjust the range as needed
    label_bins = np.arange(y_test.min(), y_test.max() + 0.2, 0.2)
    # Sum of squared errors per bin
    sums, bin_edges = np.histogram(y_test, bins=label_bins, weights=weighted_squared_errors)
    # Bin centers for plotting
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Plotting the sum of squared errors histogram
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, sums, width=np.diff(bin_edges), color='blue', edgecolor='black', align='center')
    plt.xlabel('True Label Values')
    plt.ylabel('Sum of Squared Errors')
    plt.title(title)
    plt.grid(True)

    # save the plot
    plot_filename = f"{prefix}_sum_of_squared_errors_histogram.png"
    plt.savefig(plot_filename)
    plt.close()

    return os.path.abspath(plot_filename)


def evaluate_mae(
        model: tf.keras.Model,
        X_test: Union[np.ndarray, List[np.ndarray]],
        y_test: np.ndarray,
        below_threshold: float = None,
        above_threshold: float = None,
        use_dict: bool = False) -> float:
    """
    Evaluates a given model using Mean Absolute Error (MAE) on the provided test data,
    with an option to conditionally calculate MAE based on specified thresholds.

    Parameters:
    - model (tf.keras.Model): The trained model to evaluate.
    - X_test (np.ndarray): Test features.
    - y_test (np.ndarray): True target values for the test set.
    - below_threshold (float, optional): The lower bound threshold for y_test to be included in MAE calculation.
    - above_threshold (float, optional): The upper bound threshold for y_test to be included in MAE calculation.
    - use_dict (bool, optional): Whether the model returns a dictionary with output names. Default is False.
    Returns:
    - float: The MAE loss of the model on the filtered test data.
    """
    # Make predictions
    if use_dict:
        res = model.predict(X_test)
        predictions = res['output']
    else:
        _, predictions = model.predict(X_test)

    # Process predictions
    predictions = process_predictions(predictions)
    y_test = process_predictions(y_test)

    # Filter y_test and predictions based on thresholds
    if below_threshold is not None or above_threshold is not None:
        if below_threshold is not None and above_threshold is not None:
            mask = (y_test >= above_threshold) | (y_test <= below_threshold)
        elif below_threshold is not None:
            mask = y_test <= below_threshold
        else:  # above_threshold is not None
            mask = y_test >= above_threshold

        filtered_predictions = predictions[mask]
        filtered_y_test = y_test[mask]
    else:
        filtered_predictions = predictions
        filtered_y_test = y_test

    # Calculate MAE
    mae_loss = mean_absolute_error(filtered_y_test, filtered_predictions)
    return mae_loss


def evaluate_pcc(
        model: tf.keras.Model,
        X_test: Union[np.ndarray, List[np.ndarray]],
        y_test: np.ndarray,
        logI_test: np.ndarray = None,
        logI_prev_test: np.ndarray = None,
        below_threshold: float = None,
        above_threshold: float = None,
        use_dict: bool = False) -> float:
    """
    Evaluates a given model using Pearson Correlation Coefficient (PCC) on the provided test data,
    with an option to conditionally calculate PCC based on specified thresholds.

    Parameters:
    - model (tf.keras.Model): The trained model to evaluate.
    - X_test (np.ndarray): Test features.
    - y_test (np.ndarray): True target values for the test set.
    - logI_test (np.ndarray, optional): True log intensity values. If provided along with logI_prev_test,
      PCC will be calculated between true and predicted log intensities instead of deltas.
    - logI_prev_test (np.ndarray, optional): Previous log intensity values. Required if logI_test is provided.
    - below_threshold (float, optional): The lower bound threshold for y_test to be included in PCC calculation.
    - above_threshold (float, optional): The upper bound threshold for y_test to be included in PCC calculation.
    - use_dict (bool, optional): Whether the model returns a dictionary with output names. Default is False.

    Returns:
    - float: The Pearson Correlation Coefficient between the model predictions and the filtered test data.
    """
    # Make predictions
    if use_dict:
        res = model.predict(X_test)
        predictions = res['output']
    else:
        _, predictions = model.predict(X_test)

    # Process predictions
    predictions = process_predictions(predictions)
    y_test = process_predictions(y_test)

    # Filter y_test and predictions based on thresholds
    if below_threshold is not None or above_threshold is not None:
        if below_threshold is not None and above_threshold is not None:
            mask = (y_test >= above_threshold) | (y_test <= below_threshold)
        elif below_threshold is not None:
            mask = y_test <= below_threshold
        else:  # above_threshold is not None
            mask = y_test >= above_threshold

        filtered_predictions = predictions[mask]
        filtered_y_test = y_test[mask]
        if logI_test is not None:
            logI_test = logI_test[mask]
            logI_prev_test = logI_prev_test[mask]
    else:
        filtered_predictions = predictions
        filtered_y_test = y_test

    # Calculate PCC based on logI if provided, otherwise use deltas
    if logI_test is not None and logI_prev_test is not None:
        # Calculate predicted logI by adding predicted delta to previous logI
        predicted_logI = logI_prev_test + filtered_predictions
        pcc, _ = pearsonr(logI_test.flatten(), predicted_logI.flatten())
    else:
        pcc, _ = pearsonr(filtered_y_test.flatten(), filtered_predictions.flatten())

    return pcc


def filter_ds_deprecated(
        X: np.ndarray, y: np.ndarray,
        low_threshold: float, high_threshold: float,
        N: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter and sample the dataset based on the threshold values of y with a random seed for reproducibility.

    This function creates a subset of the dataset where all samples where y is either
    below the low_threshold or above the high_threshold are included. Samples where y
    is between the low_threshold and high_threshold are randomly sampled to have a total
    of N samples in the resulting dataset, using a specified seed for random number generation.

    Parameters:
        X (np.ndarray): The input features of the dataset.
        y (np.ndarray): The output labels of the dataset.
        low_threshold (float): The lower bound threshold for selecting high delta values.
        high_threshold (float): The upper bound threshold for selecting high delta values.
        N (int): The number of samples to include from the low delta range.
        seed (int, optional): Seed for the random number generator to ensure reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The filtered and sampled input features and output labels.
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Flatten the output array to ensure mask works properly with input dimensions
    y_flat = y.flatten()

    # Mask for selecting samples where y is either too low or too high
    high_deltas_mask = (y_flat <= low_threshold) | (y_flat >= high_threshold)
    X_high_deltas = X[high_deltas_mask]
    y_high_deltas = y[high_deltas_mask]

    # Mask for selecting samples where y is in the middle range
    low_deltas_mask = (y_flat > low_threshold) & (y_flat < high_threshold)
    X_low_deltas = X[low_deltas_mask]
    y_low_deltas = y[low_deltas_mask]

    # Sample from the low deltas if they exceed the required N samples
    if len(y_low_deltas) > N:
        sampled_indices = np.random.choice(len(X_low_deltas), size=N, replace=False)
        X_low_deltas_sampled = X_low_deltas[sampled_indices]
        y_low_deltas_sampled = y_low_deltas[sampled_indices]
    else:
        X_low_deltas_sampled = X_low_deltas
        y_low_deltas_sampled = y_low_deltas

    # Combine the high delta samples and the sampled low delta samples
    X_combined = np.concatenate([X_high_deltas, X_low_deltas_sampled], axis=0)
    y_combined = np.concatenate([y_high_deltas, y_low_deltas_sampled], axis=0)

    return X_combined, y_combined


def filter_ds(
        X: np.ndarray, y: np.ndarray,
        low_threshold: float, high_threshold: float,
        N: int = 500, bins: int = 10, seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter and sample the dataset based on the threshold values of y with a random seed for reproducibility.

    This function creates a subset of the dataset where all samples where y is either
    below the low_threshold or above the high_threshold are included. Samples where y
    is between the low_threshold and high_threshold are randomly sampled within bins
    to have a total of N samples in the resulting dataset, using a specified seed for
    random number generation.

    Parameters:
        X (np.ndarray): The input features of the dataset.
        y (np.ndarray): The output labels of the dataset.
        low_threshold (float): The lower bound threshold for selecting high delta values.
        high_threshold (float): The upper bound threshold for selecting high delta values.
        N (int): The number of samples to include from the low delta range.
        bins (int): The number of bins to split the low delta range into.
        seed (int, optional): Seed for the random number generator to ensure reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The filtered and sampled input features and output labels.
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Flatten the output array to ensure mask works properly with input dimensions
    y_flat = y.flatten()

    # Create a mask to identify high delta samples (below low_threshold or above high_threshold)
    high_deltas_mask = (y_flat <= low_threshold) | (y_flat >= high_threshold)

    # Apply the high deltas mask to get the corresponding samples
    X_high_deltas = X[high_deltas_mask, :]
    y_high_deltas = y[high_deltas_mask, :]

    # Create a mask to identify low delta samples (between low_threshold and high_threshold)
    low_deltas_mask = (y_flat > low_threshold) & (y_flat < high_threshold)

    # Apply the low deltas mask to get the corresponding samples
    X_low_deltas = X[low_deltas_mask, :]
    y_low_deltas = y[low_deltas_mask, :]

    # Create bin edges for the low delta samples
    bins_edges = np.linspace(low_threshold, high_threshold, bins + 1)

    # Digitize y_low_deltas to assign each sample to a bin
    binned_indices = np.digitize(y_low_deltas.flatten(), bins_edges) - 1

    # Determine the budget per bin and remainder
    budget = N // bins
    remainder = N % bins

    # Initialize lists to store sampled low delta values
    X_low_deltas_sampled = []
    y_low_deltas_sampled = []

    # Sample low delta values from each bin
    for bin_idx in range(bins):
        # Create a mask for the current bin
        bin_mask = binned_indices == bin_idx

        # Select samples in the current bin
        X_bin = X_low_deltas[bin_mask, :]
        y_bin = y_low_deltas[bin_mask, :]

        # Determine the number of samples to draw from this bin
        bin_budget = budget + (1 if remainder > 0 else 0)
        remainder = max(0, remainder - 1)

        # Sample from the bin if it has more samples than the budget
        if len(y_bin) > bin_budget:
            sampled_indices = np.random.choice(len(X_bin), size=bin_budget, replace=False)
            X_low_deltas_sampled.append(X_bin[sampled_indices])
            y_low_deltas_sampled.append(y_bin[sampled_indices])
        else:
            # If the bin has fewer samples than the budget, include all samples
            X_low_deltas_sampled.append(X_bin)
            y_low_deltas_sampled.append(y_bin)
            remainder += bin_budget - len(y_bin)

    # Distribute the remaining budget to the center bins
    center_bins = np.arange(bins // 4, 3 * bins // 4)
    for bin_idx in center_bins:
        if remainder <= 0:
            break
        bin_mask = binned_indices == bin_idx
        X_bin = X_low_deltas[bin_mask, :]
        y_bin = y_low_deltas[bin_mask, :]

        if len(y_bin) > len(X_low_deltas_sampled[bin_idx]):
            additional_needed = min(remainder, len(y_bin) - len(X_low_deltas_sampled[bin_idx]))
            sampled_indices = np.random.choice(
                np.arange(len(y_bin)),
                size=additional_needed,
                replace=False
            )
            X_low_deltas_sampled[bin_idx] = np.concatenate((X_low_deltas_sampled[bin_idx], X_bin[sampled_indices]))
            y_low_deltas_sampled[bin_idx] = np.concatenate((y_low_deltas_sampled[bin_idx], y_bin[sampled_indices]))
            remainder -= additional_needed

    # Concatenate all sampled low delta values
    X_low_deltas_sampled = np.concatenate(X_low_deltas_sampled, axis=0)
    y_low_deltas_sampled = np.concatenate(y_low_deltas_sampled, axis=0)

    # Combine high delta samples with sampled low delta samples
    X_combined = np.concatenate([X_high_deltas, X_low_deltas_sampled], axis=0)
    y_combined = np.concatenate([y_high_deltas, y_low_deltas_sampled], axis=0)

    return X_combined, y_combined


# def evaluate_model_dummy(y_test: np.ndarray) -> float:
#     """
#     Evaluates dummy predictions (always 0) using Mean Absolute Error (MAE) on the provided test data.
#
#     Parameters:
#     - y_test (np.ndarray): True target values for the test set.
#
#     Returns:
#     - float: The MAE loss for the dummy predictions.
#     """
#     # Create dummy predictions, an array of zeros with the same shape as y_test
#     dummy_predictions = np.zeros_like(y_test)
#
#     # Assume process_predictions is a function you use to preprocess your predictions and true labels.
#     # If you have such a function, apply it here. Otherwise, you can directly calculate MAE.
#     # dummy_predictions = process_predictions(dummy_predictions)
#     # y_test = process_predictions(y_test)
#
#     # Calculate MAE
#     mae_loss = mean_absolute_error(y_test, dummy_predictions)
#
#     return mae_loss


# def evaluate_model_cond_dummy(
#         y_test: np.ndarray,
#         below_threshold: float = None,
#         above_threshold: float = None) -> float:
#     """
#     Evaluates dummy predictions (always 0) using Mean Absolute Error (MAE) on the provided test data,
#     with an option to conditionally calculate MAE based on specified thresholds.
#
#     Parameters:
#     - y_test (np.ndarray): True target values for the test set.
#     - below_threshold (float, optional): The lower bound threshold for y_test to be included in MAE calculation.
#     - above_threshold (float, optional): The upper bound threshold for y_test to be included in MAE calculation.
#
#     Returns:
#     - float: The MAE loss for the dummy predictions on the filtered test data.
#     """
#     # Create dummy predictions, an array of zeros with the same shape as y_test
#     dummy_predictions = np.zeros_like(y_test)
#
#     # Filter y_test based on thresholds
#     if below_threshold is not None and above_threshold is not None:
#         mask = (y_test >= above_threshold) | (y_test <= below_threshold)
#     elif below_threshold is not None:
#         mask = y_test <= below_threshold
#     elif above_threshold is not None:
#         mask = y_test >= above_threshold
#     else:
#         mask = np.ones_like(y_test, dtype=bool)
#
#     filtered_y_test = y_test[mask]
#     filtered_dummy_predictions = dummy_predictions[mask]  # This remains zeros but matches the filtered_y_test's shape
#
#     # Calculate MAE
#     mae_loss = mean_absolute_error(filtered_y_test, filtered_dummy_predictions)
#
#     return mae_loss


def reshape_X(
        X: np.ndarray,
        n_features_list: List[int],
        inputs_to_use: List[str],
        add_slope: bool = True,
        model_type: str = 'mlp') -> Union[ndarray, ndarray, tuple]:
    """
    Reshapes the input sep_files for the MLP, CNN or RNN model based on the model type and input dimensions.

    Parameters:
    - X (np.ndarray): The input cme_files to reshape.
    - n_features_list (List[int]): A list of input dimensions for the CNN or RNN model.
    - inputs_to_use (List[str]): The list of input types to use.
    - add_slope (bool): Whether to add slope features.
    - model_type (str): The type of model used ('cnn' or 'rnn').

    Returns:
    - np.ndarray: The reshaped input cme_files.
    """

    if model_type == 'mlp':
        return X
    elif model_type == '1dcnn' or model_type == 'gru' or model_type == 'seq':
        return prepare_seq_inputs(X, n_features_list, add_slope)
    elif model_type == "hybrid":
        return prepare_hybrid_inputs(
            X,
            tsf_input_dims=n_features_list,
            with_slope=add_slope,
            mlp_input_dim=20 + len(inputs_to_use))
    else:
        return X


def prepare_hybrid_inputs(
        data: np.ndarray,
        tsf_extractor_type: str = 'seq',
        tsf_input_dims=None,
        with_slope: bool = False,
        mlp_input_dim: int = 23) -> Tuple[np.ndarray, ...]:
    """
    Splits the input cme_files into parts for the TSF extractor branches (CNN or RNN) and a part for the MLP branch
    of the hybrid model.

    Parameters:
    - cme_files (np.ndarray): The combined input cme_files array.
    - tsf_extractor_type (str): The type of time series feature extractor ('cnn' or 'rnn').
    - tsf_input_dims (List[int]): The dimensions for each TSF extractor input.
    - with_slope (bool): Whether to add additional inputs for slopes.
    - mlp_input_dim (int): The number of features for the MLP branch.

    Returns:
    - Tuple: A tuple containing arrays for TSF extractor inputs and MLP input.
    """

    # Prepare inputs for the TSF extractor
    if tsf_extractor_type == '1dcnn' or tsf_extractor_type == 'gru' or tsf_extractor_type == 'seq':
        tsf_inputs = prepare_seq_inputs(data, tsf_input_dims, with_slope)
    else:
        raise ValueError("Invalid tsf_extractor_type. Must be 'cnn' or 'rnn'.")

    # Prepare inputs for the MLP branch
    # Assuming that the MLP input is located after the TSF input cme_files
    start_index = sum(tsf_input_dims)
    mlp_input = data[:, start_index:start_index + mlp_input_dim]

    return *tsf_inputs, mlp_input


def prepare_seq_inputs(
        data: np.ndarray,
        seq_input_dims: List[int] = None,
        with_slope: bool = False,
        use_ch: bool = True) -> Union[Tuple, ndarray]:
    """
    Prepares CNN inputs for the time series data, including regular and slope inputs if with_slope is True.


    Parameters:
    - cme_files (np.ndarray): The combined input cme_files array.
    - seq_input_dims (List[int]): The dimensions for each cnn or rnn input.
    - with_slope (bool): Whether to add additional inputs for slopes.
    - use_ch (bool): Whether to use channel-wise concatenation.


    Returns:
    - Tuple: Tuple of arrays, each for CNN inputs.
    """
    if seq_input_dims is None:
        seq_input_dims = [25, 24]  # Default dimensions for regular and slope inputs

    # Initialize a list to store CNN inputs
    seq_inputs = []

    # Split input dimensions into regular and slope inputs if with_slope is True
    if with_slope:
        half_len = len(seq_input_dims) // 2
        regular_dims = seq_input_dims[:half_len]
        slope_dims = seq_input_dims[half_len:]

        # print(f"Regular dims: {regular_dims}")
        # print(f"Slope dims: {slope_dims}")
    else:
        regular_dims = seq_input_dims
        slope_dims = []

    if use_ch:
        # Handle channel-wise concatenation
        # For regular cme_files
        regular_channels = [data[:, start:start + dim].reshape(-1, dim, 1) for start, dim in
                            zip(range(0, sum(regular_dims), regular_dims[0]), regular_dims)]
        seq_input_regular = np.concatenate(regular_channels, axis=-1) if regular_channels else None

        # print(f"Regular channels: {seq_input_regular}")
        # print(f"Regular channels shape: {seq_input_regular.shape}")

        # Add regular channels to inputs
        if seq_input_regular is not None:
            seq_inputs.append(seq_input_regular)
        # For slope cme_files, if with_slope is True
        if with_slope:
            slope_channels = [data[:, start:start + dim].reshape(-1, dim, 1) for start, dim in
                              zip(range(sum(regular_dims), sum(regular_dims) + sum(slope_dims), slope_dims[0]),
                                  slope_dims)]

            # print(f"Slope channels: {slope_channels}")

            seq_input_slope = np.concatenate(slope_channels, axis=-1) if slope_channels else None

            # print(f"Slope channels: {seq_input_slope}")
            # print(f"Slope channels shape: {seq_input_slope.shape}")

            # Add slope channels to inputs
            if seq_input_slope is not None:
                seq_inputs.append(seq_input_slope)

        # print(f"Final CNN inputs: {seq_inputs}")

        # Check if all seq_inputs have the same shape
        if all(seq_input.shape == seq_inputs[0].shape for seq_input in seq_inputs):
            # All inputs have the same shape, concatenate them along the second axis
            concatenate_seq_inputs = np.concatenate(seq_inputs, axis=-1)
            return concatenate_seq_inputs
        else:
            # The inputs do not all have the same shape, return the tuple of original inputs
            return tuple(seq_inputs)
    else:
        # Generate CNN inputs for regular cme_files
        start_index = 0
        for dim in regular_dims:
            end_index = start_index + dim
            seq_input = data[:, start_index:end_index].reshape((-1, dim, 1))
            seq_inputs.append(seq_input)
            start_index = end_index

        # Generate CNN inputs for slope cme_files if with_slope is True
        for dim in slope_dims:
            end_index = start_index + dim
            slope_input = data[:, start_index:end_index].reshape((-1, dim, 1))
            seq_inputs.append(slope_input)
            start_index = end_index

        return tuple(seq_inputs)


def find_threshold_crossing_indices(values: np.ndarray, threshold: float) -> np.ndarray:
    """
    Find indices where values cross the threshold.

    Parameters:
    - values (np.ndarray): The time series data.
    - threshold (float): The threshold to check for crossings.

    Returns:
    - np.ndarray: Indices where the values cross the threshold from below.
    """
    # Identify where values cross the threshold from below
    crossings = np.where((values[:-1] < threshold) & (values[1:] >= threshold))[0]
    return crossings + 1  # +1 to correct for the shift due to diff


# def find_shift_lag(
#         timestamps: np.ndarray,
#         actual_ts: np.ndarray,
#         predicted_ts: np.ndarray) -> float:
#     """
#     Find the shift lag by calculating the MAE for various shifts of predicted data, both left (positive shifts)
#     and right (negative shifts).
#
#     Parameters:
#     - timestamps (np.ndarray): Timestamps for the time series data.
#     - actual_ts (np.ndarray): The actual time series data.
#     - predicted_ts (np.ndarray): The predicted time series data.
#
#     Returns:
#     - float: The shift lag in minutes for the best alignment of predicted data to actual data. Positive lag indicates
#       that the predicted series should be shifted earlier, and negative lag indicates a shift to later times.
#     """
#     min_mae = float('inf')
#     best_shift = 0
#     time_interval = (timestamps[1] - timestamps[0]) / np.timedelta64(1, 'm')  # Assuming uniform spacing
#
#     # Consider shifts up to a reasonable limit to avoid overfitting to noise
#     max_shifts = min(len(actual_ts) // 2, 12)  # For example, limit to 10 shifts ( if data is 5-minute intervals)
#
#     # Loop for shifting left (positive shifts)
#     for shift in range(1, max_shifts + 1):
#         shifted_predicted_ts = np.roll(predicted_ts, -shift)
#         mae = np.mean(np.abs(actual_ts[shift:] - shifted_predicted_ts[:-shift]))
#         if mae < min_mae:
#             min_mae = mae
#             best_shift = shift
#
#     # Loop for shifting right (negative shifts)
#     for shift in range(1, max_shifts + 1):
#         shifted_predicted_ts = np.roll(predicted_ts, shift)
#         mae = np.mean(np.abs(actual_ts[:-shift] - shifted_predicted_ts[shift:]))
#         if mae < min_mae:
#             min_mae = mae
#             best_shift = -shift
#
#     # Convert the best shift to minutes
#     shift_lag = best_shift * time_interval
#     return shift_lag


def find_shift_lag(
        timestamps: np.ndarray,
        actual_ts: np.ndarray,
        predicted_ts: np.ndarray,
        smoothing_window: int) -> float:
    """
    Find the shift lag focusing on the onset to peak period by calculating the MAE for various shifts
    of predicted data, considering the smoothed slope for onset detection.

    Parameters:
    - timestamps (np.ndarray): Timestamps for the time series data, 5 minutes apart.
    - actual_ts (np.ndarray): The actual time series data.
    - predicted_ts (np.ndarray): The predicted time series data.
    - smoothing_window (int): The number of data points used for smoothing to detect the onset (1 hour equivalent).

    Returns:
    - float: The shift lag in minutes for the best alignment of predicted data to actual data from onset to peak.
             Positive lag indicates that the predicted series should be shifted earlier, and negative lag
             indicates a shift to later times.
    """
    # Convert smoothing window to the equivalent in terms of data points, given data points are 5 minutes apart
    # Here, smoothing_window is expected in terms of hours, so convert to the number of 5-minute intervals
    points_per_hour = 12  # 60 minutes / 5 minutes
    smoothing_points = smoothing_window * points_per_hour

    def smooth_data(data: np.ndarray) -> np.ndarray:
        """
        Smooths the data using a simple moving average over a specified window.
        """
        return np.convolve(data, np.ones(smoothing_points) / smoothing_points, mode='valid')

    def find_onset_peak(data: np.ndarray) -> (int, int):
        """
        Finds the onset and peak of an event within the data based on slope analysis after smoothing.
        """
        smoothed_data = smooth_data(data)
        slopes = np.diff(smoothed_data)

        # Find the first positive slope as onset and the max value as peak
        onset_idx = np.where(slopes > 0)[0][0]
        peak_idx = np.argmax(smoothed_data) + (smoothing_points // 2)  # Adjusting for the convolution effect

        return onset_idx, peak_idx

    onset_actual, peak_actual = find_onset_peak(actual_ts)
    onset_predicted, peak_predicted = find_onset_peak(predicted_ts)

    min_mae = float('inf')
    best_shift = 0
    for shift in range(-len(predicted_ts), len(predicted_ts)):
        shifted_predicted_ts = np.roll(predicted_ts, shift)

        # Adjusting calculation to ensure positive/negative lag interpretation is correct
        onset_adjusted = max(0, onset_actual - shift) if shift < 0 else onset_actual
        peak_adjusted = min(peak_actual, peak_actual - shift) if shift < 0 else peak_actual

        valid_range_actual = slice(onset_adjusted, peak_adjusted)
        valid_range_predicted = slice(onset_adjusted + shift, peak_adjusted + shift)

        if valid_range_actual.stop - valid_range_actual.start > 0:
            mae = np.mean(np.abs(actual_ts[valid_range_actual] - shifted_predicted_ts[valid_range_predicted]))
            if mae < min_mae:
                min_mae = mae
                best_shift = shift

    # Correcting the calculation of time shift in minutes
    shift_lag = best_shift * 5  # Convert shift index to minutes considering 5-minute intervals
    return shift_lag


def find_shift_lag_with_correlation(timestamps: np.ndarray, actual_ts: np.ndarray, predicted_ts: np.ndarray) -> float:
    """
    Find the shift lag by calculating the correlation for various shifts of predicted data, both left and right.

    Parameters:
    - timestamps (np.ndarray): Timestamps for the time series data.
    - actual_ts (np.ndarray): The actual time series data.
    - predicted_ts (np.ndarray): The predicted time series data.

    Returns:
    - float: The shift lag in minutes for the best alignment of predicted data to actual data. Positive values
      indicate the predicted series should be shifted left (earlier), and negative values indicate a shift
      right (later).
    """
    # Ensure both series have the same length
    len_diff = len(actual_ts) - len(predicted_ts)
    if len_diff > 0:
        predicted_ts = np.pad(predicted_ts, (len_diff, 0), 'constant', constant_values=0)
    elif len_diff < 0:
        actual_ts = np.pad(actual_ts, (abs(len_diff), 0), 'constant', constant_values=0)

    # Calculate correlation and lags
    correlation = correlate(actual_ts, predicted_ts, mode='full')
    lags = correlation_lags(len(actual_ts), len(predicted_ts), mode='full')
    time_interval = (timestamps[1] - timestamps[0]) / np.timedelta64(1, 'm')  # assuming uniform spacing

    # Find the lag with the maximum correlation
    max_corr_index = np.argmax(correlation)
    optimal_lag = lags[max_corr_index]

    # Convert the optimal lag to minutes
    shift_lag = -optimal_lag * time_interval
    return shift_lag


# def pearson_correlation_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
#     """
#     Calculate the Pearson Correlation Coefficient (PCC) using native TensorFlow operations.
#
#     This function computes the PCC between two tensors using the formula:
#     PCC = cov(X, Y) / (std(X) * std(Y))
#
#     Where:
#     - cov(X, Y) is the covariance between X and Y
#     - std(X) and std(Y) are the standard deviations of X and Y respectively
#
#     :param y_true: Ground truth values. Shape: [batch_size, ...].
#     :param y_pred: Predicted values. Shape: [batch_size, ...].
#     :return: Pearson Correlation Coefficient. Shape: scalar.
#     """
#     # Flatten input tensors to 1D
#     # Calculate PCC
#     y_true_flat = tf.keras.backend.flatten(y_true)  # Flatten to 1D vector
#     y_pred_flat = tf.keras.backend.flatten(y_pred)  # Flatten to 1D vector
#
#     # Calculate means
#     y_true_mean = tf.reduce_mean(y_true_flat)
#     y_pred_mean = tf.reduce_mean(y_pred_flat)
#
#     # Center the data (subtract mean)
#     y_true_centered = y_true_flat - y_true_mean
#     y_pred_centered = y_pred_flat - y_pred_mean
#
#     # Calculate covariance
#     covariance = tf.reduce_mean(y_true_centered * y_pred_centered)
#
#     # Calculate standard deviations
#     y_true_std = tf.math.reduce_std(y_true_flat)
#     y_pred_std = tf.math.reduce_std(y_pred_flat)
#
#     # Calculate PCC
#     pcc = covariance / (y_true_std * y_pred_std)
#
#     return pcc

def pearson_correlation_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None) -> tf.Tensor:
    """
    Calculate the Pearson Correlation Coefficient (PCC) using native TensorFlow operations,
    with support for sample weights.

    This function computes the PCC between two tensors using the formula:
    PCC = cov(X, Y) / (std(X) * std(Y))

    Where:
    - cov(X, Y) is the covariance between X and Y
    - std(X) and std(Y) are the standard deviations of X and Y respectively

    :param y_true: Ground truth values. Shape: [batch_size, ...].
    :param y_pred: Predicted values. Shape: [batch_size, ...].
    :param sample_weight: Optional tensor of sample weights. Shape: [batch_size, ...].
    :return: Pearson Correlation Coefficient. Shape: scalar.
    """

    # Ensure consistent data type (float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, tf.float32)

    # Flatten input tensors to 1D
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_pred)

    if sample_weight is not None:
        sample_weight = tf.keras.backend.flatten(sample_weight)
        sample_weight /= tf.reduce_sum(sample_weight)  # Normalize the weights

    # Calculate weighted means
    if sample_weight is not None:
        y_true_mean = tf.reduce_sum(y_true_flat * sample_weight)
        y_pred_mean = tf.reduce_sum(y_pred_flat * sample_weight)
    else:
        y_true_mean = tf.reduce_mean(y_true_flat)
        y_pred_mean = tf.reduce_mean(y_pred_flat)

    # Center the data (subtract mean)
    y_true_centered = y_true_flat - y_true_mean
    y_pred_centered = y_pred_flat - y_pred_mean

    # Calculate weighted covariance
    if sample_weight is not None:
        covariance = tf.reduce_sum(y_true_centered * y_pred_centered * sample_weight)
    else:
        covariance = tf.reduce_mean(y_true_centered * y_pred_centered)

    # Calculate weighted standard deviations
    if sample_weight is not None:
        y_true_var = tf.reduce_sum(tf.square(y_true_centered) * sample_weight)
        y_pred_var = tf.reduce_sum(tf.square(y_pred_centered) * sample_weight)
    else:
        y_true_var = tf.reduce_mean(tf.square(y_true_centered))
        y_pred_var = tf.reduce_mean(tf.square(y_pred_centered))

    y_true_std = tf.sqrt(y_true_var)
    y_pred_std = tf.sqrt(y_pred_var)

    # Calculate PCC
    pcc = covariance / (y_true_std * y_pred_std)

    return pcc


def evaluate_lag_error(
        timestamps: np.ndarray,
        actual_ts: np.ndarray,
        predicted_ts: np.ndarray,
        threshold: float) -> Tuple[float, float, float]:
    """
    Evaluates the lag error for threshold crossings and the shift lag for the best alignment of predicted data to actual
    data. Also computes the average lag for the threshold crossings.
    TODO: fix lag

    Parameters:
    - timestamps (np.ndarray): Timestamps for the time series data.
    - actual_ts (np.ndarray): The actual time series data.
    - predicted_ts (np.ndarray): The predicted time series data.
    - threshold (float): The threshold for identifying significant events in the time series.

    Returns:
    - float: The lag error in minutes for the threshold crossings.
    - float: The shift lag in minutes for the best alignment of predicted data to actual data.
    - float: The average lag in minutes for the threshold crossings.
    """

    # Part 1: threshold lag
    global threshold_lag, shift_lag, avg_lag
    # Convert pandas.Series to numpy.ndarray if necessary
    if isinstance(timestamps, pd.Series):
        timestamps = timestamps.to_numpy()
    if isinstance(actual_ts, pd.Series):
        actual_ts = actual_ts.to_numpy()
    if isinstance(predicted_ts, pd.Series):
        predicted_ts = predicted_ts.to_numpy()

    # print(f"Timestamps: {timestamps}, of type {type(timestamps)}")
    # print(f"Actual TS: {actual_ts}, of type {type(actual_ts)}")
    # print(f"Predicted TS: {predicted_ts}, of type {type(predicted_ts)}")
    # print(f"Threshold: {threshold}")
    # Find threshold crossings in actual and predicted time series
    actual_crossings = find_threshold_crossing_indices(actual_ts, threshold)
    predicted_crossings = find_threshold_crossing_indices(predicted_ts, threshold)

    # print(f"Actual Crossings: {actual_crossings}, of type {type(actual_crossings)}")
    # print(f"Predicted Crossings: {predicted_crossings}, of type {type(predicted_crossings)}")

    # Ensure there is at least one crossing in both actual and predicted to compute lag
    if len(actual_crossings) == 0 or len(predicted_crossings) == 0:
        if len(actual_crossings) == 0:
            print("No crossings found in actual time series.")
        if len(predicted_crossings) == 0:
            print("No crossings found in predicted time series.")

        threshold_lag = np.nan
    else:
        # Compute the lag for the first crossing as an example
        # More sophisticated logic can be applied to handle multiple crossings
        first_actual_crossing_time = timestamps[actual_crossings[0]]
        first_predicted_crossing_time = timestamps[predicted_crossings[0]]
        threshold_lag = first_predicted_crossing_time - first_actual_crossing_time
        # convert lag from nanoseconds to minutes
        threshold_lag = threshold_lag / np.timedelta64(1, 'm')

    # Part 2: shift lag
    # Calculate shift lag
    # shift_lag = find_shift_lag(timestamps, actual_ts, predicted_ts)
    shift_lag = find_shift_lag_with_correlation(timestamps, actual_ts, predicted_ts)
    # average lag
    avg_lag = (threshold_lag + shift_lag) / 2

    return threshold_lag, shift_lag, avg_lag


def asymmetric_weight_silu(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Asymmetric weight based on SiLU function: AW = 1 + SiLU(y_true - y_pred)
    Args:
    - y_true (tf.Tensor): Ground truth labels.
    - y_pred (tf.Tensor): Predicted labels.

    Returns:
    - tf.Tensor: The asymmetric weight tensor.
    """
    diff = y_true - y_pred
    silu = diff / (1 + tf.exp(-diff))  # SiLU function
    return 1 + silu


def asymmetric_weight_sigmoid(y_true: tf.Tensor, y_pred: tf.Tensor, eta: float = 0.8, zeta: float = 2) -> tf.Tensor:
    """
    Asymmetric weight based on Sigmoid function:
    AW = 2(1 - eta) * Sigmoid(zeta * (y_true - y_pred)) + eta
    Args:
    - y_true (tf.Tensor): Ground truth labels.
    - y_pred (tf.Tensor): Predicted labels.
    - eta (float, optional): percentage parameter that determines the error for the overestimation of the target.
    - zeta (float, optional): Controls the steepness of the Sigmoid function.

    Returns:
    - tf.Tensor: The asymmetric weight tensor.
    """
    diff = y_true - y_pred
    sigmoid = tf.nn.sigmoid(zeta * diff)  # Sigmoid with zeta controlling the steepness
    return 2 * (1 - eta) * sigmoid + eta


def coreg(y_true, y_pred, pcc_weights=None):
    """
    Correlation based regularizer:
    Compute 1 minus the Pearson Correlation Coefficient (PCC) between two sets of predictions.
    
    PCC measures the linear correlation between two variables, providing a value 
    between -1 (perfect negative correlation) and 1 (perfect positive correlation).
    A value of 0 indicates no linear correlation.
    
    Returns 1-PCC which can be used as a loss function, where 0 represents perfect 
    positive correlation and 2 represents perfect negative correlation.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth values.
    y_pred : tf.Tensor
        Predicted values.
    pcc_weights : tf.Tensor, optional
        Weights for each observation. If None, uniform weights are used.

    Returns
    -------
    tf.Tensor
        A scalar tensor representing 1-PCC (for use as a loss function)
    """
    if pcc_weights is None:
        pcc_weights = tf.ones_like(y_true)

    # Center the data by subtracting their means
    y_true_centered = y_true - tf.reduce_mean(y_true)
    y_pred_centered = y_pred - tf.reduce_mean(y_pred)

    # Compute covariance
    cov = tf.reduce_sum(pcc_weights * y_true_centered * y_pred_centered)

    # Compute standard deviations
    std_y_true = tf.sqrt(tf.reduce_sum(pcc_weights * tf.square(y_true_centered)))
    std_y_pred = tf.sqrt(tf.reduce_sum(pcc_weights * tf.square(y_pred_centered)))

    # Compute PCC
    pcc = cov / (std_y_true * std_y_pred + K.epsilon())

    # Return 1-PCC
    return 1.0 - pcc


def cmse(
        y_true: tf.Tensor, y_pred: tf.Tensor,
        lambda_factor: float,
        phase_manager: 'TrainingPhaseManager',
        train_mse_weight_dict: Optional[Dict[float, float]] = None,
        val_mse_weight_dict: Optional[Dict[float, float]] = None,
        train_pcc_weight_dict: Optional[Dict[float, float]] = None,
        val_pcc_weight_dict: Optional[Dict[float, float]] = None,
        asym_type: Optional[str] = None,  # New parameter to choose asymmetric weight type
) -> tf.Tensor:
    """
    Custom loss function combining Mean Squared Error (MSE) and Pearson Correlation Coefficient (PCC)
    with re-weighting based on label values. The final loss is a combination of weighted MSE and
    weighted PCC with a scaling factor lambda_factor.

    Args:
    - y_true (tf.Tensor): Ground truth labels.
    - y_pred (tf.Tensor): Predicted labels.
    - lambda_factor (float): Scaling factor for the PCC portion of the loss.
    - phase_manager (TrainingPhaseManager): Manager that tracks whether we are in training or validation phase.
    - train_mse_weight_dict (dict, optional): Dictionary mapping label values to weights for training MSE samples.
    - val_mse_weight_dict (dict, optional): Dictionary mapping label values to weights for validation MSE samples.
    - train_pcc_weight_dict (dict, optional): Dictionary mapping label values to weights for training PCC samples.
    - val_pcc_weight_dict (dict, optional): Dictionary mapping label values to weights for validation PCC samples.
    - asym_type (str, optional): Type of asymmetric weight to use ('silu' or 'sigmoid').

    Returns:
    - tf.Tensor: The calculated loss value as a single scalar.
    """
    # Select the appropriate weight dictionaries based on the mode
    mse_weight_dict = train_mse_weight_dict if phase_manager.is_training_phase() else val_mse_weight_dict
    pcc_weight_dict = train_pcc_weight_dict if phase_manager.is_training_phase() else val_pcc_weight_dict

    # Generate the weight tensors for MSE and PCC using the optimized function
    mse_weights = create_weight_tensor_fast(y_true, mse_weight_dict)
    pcc_weights = create_weight_tensor_fast(y_true, pcc_weight_dict)

    # Apply asymmetric weight if specified
    if asym_type == 'silu':
        asym_weights = asymmetric_weight_silu(y_true, y_pred)
    elif asym_type == 'sigmoid':
        asym_weights = asymmetric_weight_sigmoid(y_true, y_pred)
    else:
        asym_weights = 1.0  # No asymmetric weighting if not specified

    # Compute the Mean Squared Error (MSE) with asymmetric weights
    mse = tf.reduce_mean(asym_weights * mse_weights * tf.square(y_pred - y_true))

    # Compute the correlation regularization term using coreg
    pcc_loss = coreg(y_true, y_pred, pcc_weights)

    # Combine the weighted MSE and weighted PCC with lambda_factor
    loss = mse + lambda_factor * pcc_loss

    # Return the final loss as a single scalar value
    return loss


def nz_hump(x: tf.Tensor, steepness: float = 64.0) -> tf.Tensor:
    """
    Implements a double sigmoid function of the form:
    y = 1/(1+exp(steepness*(x-0.4))) + 1/(1+exp(steepness*(-x-0.4))) - 1. 
    
    this is a for the near-zero hump.
    
    This creates a symmetric function with two sigmoid transitions centered at x=±0.4.
    The steepness parameter controls how sharp the transitions are.

    Args:
        x (tf.Tensor): Input tensor
        steepness (float): Controls the steepness of the sigmoid transitions. 
                          Default is 999.0 for very sharp transitions.
                          Use 1.0 for normal sigmoid steepness.

    Returns:
        tf.Tensor: Result of the double sigmoid function
    """
    left_sigmoid = 1.0 / (1.0 + tf.exp(steepness * (x - 0.4)))
    right_sigmoid = 1.0 / (1.0 + tf.exp(steepness * (-x - 0.4)))
    return left_sigmoid + right_sigmoid - 1.0


def pn_staircase(x: tf.Tensor, steepness: float = 64.0) -> tf.Tensor:
    """
    Implements a double sigmoid function of the form:
    y = 1/(1+exp(steepness*(-x+0.4))) + 1/(1+exp(steepness*(-x-0.4))) - 1

    this is for the positive and negative staircase
    
    This creates a staircase-like function with two sigmoid transitions.
    The steepness parameter controls how sharp the transitions are.

    Args:
        x (tf.Tensor): Input tensor
        steepness (float): Controls the steepness of the sigmoid transitions.
                          Default is 64.0 for sharp transitions.

    Returns:
        tf.Tensor: Result of the double sigmoid function
    """
    left_sigmoid = 1.0 / (1.0 + tf.exp(steepness * (-x + 0.4)))
    right_sigmoid = 1.0 / (1.0 + tf.exp(steepness * (-x - 0.4)))
    return left_sigmoid + right_sigmoid - 1.0


def pn_nz_loss(
        y_true: Tuple[tf.Tensor, tf.Tensor], y_pred: tf.Tensor,
        phase_manager: 'TrainingPhaseManager',
        loss_weights: Dict[str, float] = {'ce': 0.5, 'pn': 1.0, 'nz': 1.0},
        train_ce_weight_dict: Optional[Dict[float, float]] = None,
        val_ce_weight_dict: Optional[Dict[float, float]] = None,
        train_pcc_weight_dict: Optional[Dict[float, float]] = None,
        val_pcc_weight_dict: Optional[Dict[float, float]] = None,
) -> tf.Tensor:
    """
    Custom loss function combining Cross Entropy (CE) and Pearson Correlation Coefficient (PCC)
    with re-weighting based on the delta values. The final loss is a combination of weighted CE and
    weighted PCC with scaling factors specified in loss_weights. K is a constant that upper bounds the 
    absolute value of the delta values in the zero delta term.

    Args:
    - y_true (Tuple[tf.Tensor, tf.Tensor]): A tuple (y_classes, delta_batch)
        - y_classes: Ground truth class probabilities (one-hot vector)
        - delta_batch: Corresponding delta values for each sample
    - y_pred (tf.Tensor): Predicted probabilities from the model
    - loss_weights (Dict[str, float]): Dictionary containing weights for different loss components:
        - 'ce': weight for cross entropy loss
        - 'pn': weight for plus-minus delta PCC loss
        - 'nz': weight for near-zero delta PCC loss
    - phase_manager (TrainingPhaseManager): Manager that tracks whether we are in training or validation phase.
    - train_ce_weight_dict (dict, optional): Dictionary mapping label values to weights for training CE samples.
    - val_ce_weight_dict (dict, optional): Dictionary mapping label values to weights for validation CE samples.
    - train_pcc_weight_dict (dict, optional): Dictionary mapping label values to weights for training PCC samples.
    - val_pcc_weight_dict (dict, optional): Dictionary mapping label values to weights for validation PCC samples.

    Returns:
    - tf.Tensor: The calculated loss value as a single scalar.
    """

    # Get y_classes and delta_batch from y_true
    # y_true shape is (batch_size, num_classes + 1) where last column is delta
    y_classes = y_true[:, :-1]  # All columns except last
    delta_batch = y_true[:, -1]  # Last column

    # Determine which weight dictionaries to use
    ce_weight_dict = train_ce_weight_dict if phase_manager.is_training_phase() else val_ce_weight_dict
    pcc_weight_dict = train_pcc_weight_dict if phase_manager.is_training_phase() else val_pcc_weight_dict

    # Create weight tensors based on delta_batch
    ce_weights = create_weight_tensor_fast(delta_batch, ce_weight_dict)
    pcc_weights = create_weight_tensor_fast(delta_batch, pcc_weight_dict)

    # Compute Cross Entropy
    ce = -tf.reduce_mean(ce_weights * tf.reduce_sum(y_classes * tf.math.log(y_pred + K.epsilon()), axis=-1))

    # PCC loss for positive-negative delta
    # Assuming positive class index=0 and negative class index=2
    # p(p|x) - p(n|x)
    # pcc_pn = coreg(delta_batch, y_pred[:, 0] - y_pred[:, 2], pcc_weights)
    pcc_pn = coreg(pn_staircase(delta_batch), y_pred[:, 0] - y_pred[:, 2], pcc_weights)
    # PCC loss for zero delta
    # Assuming zero class index=1, and using a Gaussian kernel
    # p(z|x)
    pcc_nz = coreg(nz_hump(delta_batch), y_pred[:, 1], pcc_weights)

    loss = loss_weights['ce'] * ce + loss_weights['pn'] * pcc_pn + loss_weights['nz'] * pcc_nz
    return loss


def pcc_ce(
        y_true: Tuple[tf.Tensor, tf.Tensor], y_pred: tf.Tensor,
        phase_manager: 'TrainingPhaseManager',
        lambda_ce: float = 0.5,
        train_ce_weight_dict: Optional[Dict[float, float]] = None,
        val_ce_weight_dict: Optional[Dict[float, float]] = None,
        train_pcc_weight_dict: Optional[Dict[float, float]] = None,
        val_pcc_weight_dict: Optional[Dict[float, float]] = None,
) -> tf.Tensor:
    """
    Custom loss function combining Cross Entropy (CE) and Pearson Correlation Coefficient (PCC)
    with re-weighting based on the delta values. The final loss is a weighted sum of CE and PCC:
    loss = pcc_pn + lambda_ce * ce
    where pcc_pn is the PCC between delta values and the difference in predicted probabilities 
    for positive and negative classes.

    Args:
    - y_true (Tuple[tf.Tensor, tf.Tensor]): A tuple (y_classes, delta_batch)
        - y_classes: Ground truth class probabilities (one-hot vector)
        - delta_batch: Corresponding delta values for each sample
    - y_pred (tf.Tensor): Predicted probabilities from the model
    - lambda_ce (float): Scaling factor for the cross entropy loss. Default is 0.5.
    - phase_manager (TrainingPhaseManager): Manager that tracks whether we are in training or validation phase.
    - train_ce_weight_dict (dict, optional): Dictionary mapping label values to weights for training CE samples.
    - val_ce_weight_dict (dict, optional): Dictionary mapping label values to weights for validation CE samples.
    - train_pcc_weight_dict (dict, optional): Dictionary mapping label values to weights for training PCC samples.
    - val_pcc_weight_dict (dict, optional): Dictionary mapping label values to weights for validation PCC samples.

    Returns:
    - tf.Tensor: The calculated loss value as a single scalar.
    """

    # Get y_classes and delta_batch from y_true
    # y_true shape is (batch_size, num_classes + 1) where last column is delta
    y_classes = y_true[:, :-1]  # All columns except last
    delta_batch = y_true[:, -1]  # Last column

    # Determine which weight dictionaries to use
    ce_weight_dict = train_ce_weight_dict if phase_manager.is_training_phase() else val_ce_weight_dict
    pcc_weight_dict = train_pcc_weight_dict if phase_manager.is_training_phase() else val_pcc_weight_dict

    # Create weight tensors based on delta_batch
    ce_weights = create_weight_tensor_fast(delta_batch, ce_weight_dict)
    pcc_weights = create_weight_tensor_fast(delta_batch, pcc_weight_dict)

    # Compute Cross Entropy
    ce = -tf.reduce_mean(ce_weights * tf.reduce_sum(y_classes * tf.math.log(y_pred + K.epsilon()), axis=-1))

    # PCC loss for positive-negative delta
    # Assuming positive class index=0 and negative class index=2
    # p(p|x) - p(n|x)
    pcc_pn = coreg(delta_batch, y_pred[:, 0] - y_pred[:, 2], pcc_weights)

    loss = pcc_pn + lambda_ce * ce
    return loss


def pcc_loss(y_true: tf.Tensor, y_pred: tf.Tensor,
             phase_manager: TrainingPhaseManager,
             train_weight_dict: Optional[Dict[int, float]] = None,
             val_weight_dict: Optional[Dict[int, float]] = None) -> tf.Tensor:
    """
    Custom loss function based on the Pearson Correlation Coefficient (PCC),
    with re-weighting based on label values. The final loss is 1 - PCC.

    Args:
        y_true (tf.Tensor): Ground truth labels.
        y_pred (tf.Tensor): Predicted labels.
        phase_manager (TrainingPhaseManager): Manager that tracks whether we are in training or validation phase.
        train_weight_dict (dict, optional): Dictionary mapping label values to weights for training samples.
        val_weight_dict (dict, optional): Dictionary mapping label values to weights for validation samples.

    Returns:
        tf.Tensor: The calculated loss value as a single scalar.
    """
    weight_dict = train_weight_dict if phase_manager.is_training_phase() else val_weight_dict

    # Generate the weight tensor using the optimized function
    weights = create_weight_tensor_fast(y_true, weight_dict)

    # Ensure y_true and y_pred are of type float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Use coreg function to calculate loss
    return coreg(y_true, y_pred, weights)


def get_loss(loss_key: str = 'mse', lambda_factor: float = 3.3, norm_factor: float = 1) -> Callable[
    [tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Given the key, return the appropriate loss function for the model.

    :param loss_key: Key for the loss function.
    :param lambda_factor: Weighting factor for the PCC term when using 'mse_pcc' loss. Default is 0.5.
    :param norm_factor: Normalization factor for the PCC term when using 'mse_pcc' loss. Default is None.

    :return: Loss function for TensorFlow model compilation.


    """

    if loss_key == 'mse':
        return tf.keras.losses.MeanSquaredError()

    else:

        raise ValueError(f"Unknown loss key: {loss_key}")


def train_step(
        model: tf.keras.Model,
        X: tf.Tensor,
        y: tf.Tensor,
        sample_weights: Optional[tf.Tensor],
        loss_fn: Callable[[tf.Tensor, tf.Tensor, Optional[tf.Tensor]], tf.Tensor],
        optimizer: Optimizer,
        output_key: int
) -> tf.Tensor:
    """
    Executes a single training step for the specified model output.

    :param model: The model to train.
    :param X: Input data.
    :param y: True labels for the specified output.
    :param sample_weights: Optional sample weights for loss computation.
    :param loss_fn: Loss function that takes true labels, predictions, and sample weights.
    :param optimizer: Optimizer to use for the training step.
    :param output_key: The key for the specific model output to train on.
    :return: The computed loss for the current training step.
    """
    with tf.GradientTape() as tape:
        # Forward pass: Compute predictions for the specified output.
        y_pred = model(X, training=True)[output_key]
        # Compute the loss using the loss function.
        loss = loss_fn(y, y_pred, sample_weights)

    # Compute gradients of the loss with respect to model parameters.
    gradients = tape.gradient(loss, model.trainable_variables)
    # Apply the gradients to update the model's parameters.
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def custom_train_loop(
        model: tf.keras.Model,
        X_train: Union[tf.Tensor, np.array],
        y_train: Union[tf.Tensor, np.array],
        train_weights: Optional[tf.Tensor],
        X_val: Optional[Union[tf.Tensor, np.array]],
        y_val: Optional[Union[tf.Tensor, np.array]],
        val_weights: Optional[tf.Tensor],
        loss_fn: Callable,
        optimizer: Optimizer,
        epochs: int,
        batch_size: int,
        output_key: int = 1,
        callbacks: Optional[List[Callback]] = None,
        verbose: int = 1
) -> Dict[str, List[float]]:
    """
    Custom training loop for a specific model output that mimics the behavior of TensorFlow's `fit` method,
    with support for callbacks, sample weights, and optional validation. Returns history of training.

    :param model: The model to train.
    :param X_train: Training input data.
    :param y_train: Training labels for the specified output.
    :param train_weights: Optional sample weights for training data.
    :param X_val: Optional validation input data.
    :param y_val: Optional validation labels for the specified output.
    :param val_weights: Optional sample weights for validation data.
    :param loss_fn: Custom loss function that accepts sample weights.
    :param optimizer: Optimizer to use for training.
    :param epochs: Number of epochs to train.
    :param batch_size: Size of the batches for training.
    :param output_key: The key for the specific model output to train on.
    :param callbacks: List of callbacks to apply during training.
    :param verbose: Verbosity mode (0 = silent, 1 = progress bar).
    :return: Dictionary containing lists of loss and validation loss per epoch.
    """
    # Initialize callback list and set model stop_training attribute
    if callbacks is None:
        callbacks = []
    for callback in callbacks:
        callback.set_model(model)
        callback.on_train_begin()

    num_samples = X_train.shape[0]
    steps_per_epoch = num_samples // batch_size

    # History dictionary to store loss and validation loss
    history = {'loss': [], 'val_loss': []} if X_val is not None else {'loss': []}

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0

        # Training loop
        for step in range(steps_per_epoch):
            # Generate mini-batch indices
            batch_start = step * batch_size
            batch_end = (step + 1) * batch_size

            # Extract mini-batch data
            X_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]
            batch_weights = train_weights[batch_start:batch_end] if train_weights is not None else None

            # Perform a training step
            batch_loss = train_step(model, X_batch, y_batch, batch_weights, loss_fn, optimizer, output_key)
            epoch_loss += batch_loss.numpy()

            # Callbacks at the end of each batch
            for callback in callbacks:
                callback.on_batch_end(step, {'loss': batch_loss.numpy()})

        # Average loss over the epoch
        epoch_loss /= steps_per_epoch

        # Store the epoch loss in the history
        history['loss'].append(epoch_loss)

        # Validation at the end of the epoch if validation data is provided
        if X_val is not None and y_val is not None:
            val_pred = model(X_val, training=False)[output_key]
            val_loss = loss_fn(y_val, val_pred, val_weights).numpy()
            history['val_loss'].append(val_loss)

            # Print progress if verbose
            if verbose:
                print(f"loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")

            # Callbacks at the end of each epoch
            for callback in callbacks:
                callback.on_epoch_end(epoch, {'loss': epoch_loss, 'val_loss': val_loss})
        else:
            # Print progress if verbose and no validation
            if verbose:
                print(f"loss: {epoch_loss:.4f}")

            # Callbacks at the end of each epoch when no validation data is provided
            for callback in callbacks:
                callback.on_epoch_end(epoch, {'loss': epoch_loss})

        # Early stopping check
        if model.stop_training:
            print("Early stopping...")
            break

    # Callbacks at the end of training
    for callback in callbacks:
        callback.on_train_end()

    return history


class PrintBatchMSE(Callback):
    def on_batch_end(self, batch, logs=None):
        # Check if logs dictionary is not None
        if logs is not None:
            # Print batch MSE
            batch_mse = logs.get('loss')
            print(f"Batch {batch + 1}, MSE: {batch_mse:.4f}")


def compute_sample_weights(y_train, num_bins=100):
    """
    Compute sample weights based on the frequency of target values.

    Parameters:
    - y_train: np.ndarray, the training target values.
    - num_bins: int, the number of bins to use for histogram.

    Returns:
    - weights: np.ndarray, the computed weights for each sample in y_train.
    """
    # Compute histogram
    hist, bin_edges = np.histogram(y_train, bins=num_bins)

    # Compute bin indexes for each sample
    bin_indexes = np.digitize(y_train, bin_edges[:-1], right=True)

    # Compute weights as inverse frequency
    weights = 1.0 / hist[bin_indexes - 1]  # Subtract 1 because bins are 1-indexed in digitize

    # Normalize weights to make the least frequent class have a weight of 1
    weights /= weights.min()

    return weights


def build_dataset_from_numpy(x, y, batch_size, options=None):
    """
    Builds a tf.data.Dataset from NumPy arrays with optional sharding.

    Args:
        x (np.ndarray): Input features array.
        y (np.ndarray): Labels array.
        batch_size (int): Global batch size to be used for the dataset. This is the total batch size that gets
                          divided across all replicas (workers/GPUs).
        options (tf.data.Options, optional): Dataset options to apply. Use this to specify sharding and other
                                             behaviors. Defaults to None, in which case the default options are used.

    Returns:
        tf.data.Dataset: A TensorFlow Dataset object ready for training or evaluation.
    """
    # Create a dataset from the input features and labels
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    # Shuffle the dataset with a buffer size equal to the length of the dataset
    dataset = dataset.shuffle(buffer_size=len(x))
    # Batch the dataset with the provided global batch size
    dataset = dataset.batch(batch_size)

    # If options are provided, apply them to the dataset
    if options:
        dataset = dataset.with_options(options)

    return dataset


def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set TensorFlow to use deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '0'
