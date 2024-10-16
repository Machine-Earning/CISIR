import os
import random
import traceback
from collections import Counter
from typing import Tuple, List, Optional, Union, Callable, Dict, Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.lines import Line2D
from numpy import ndarray
from scipy import stats
from scipy.signal import correlate, correlation_lags
from scipy.stats import pearsonr
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
    Softmax,
    Multiply,
    Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.regularizers import l2

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


def create_1dcnn(
        input_dims: list,
        hiddens: List[tuple],
        repr_dim: int = 50,
        output_dim: int = 1,
        pds: bool = False,
        l2_reg: float = None,
        dropout_rate: float = 0.0,
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
    - repr_dim (int): The number of units in the fully connected layer. Default is 50.
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

            if dropout_rate > 0.0:
                x = Dropout(dropout_rate)(x)

            if pool_type == 'max':
                x = MaxPooling1D(pool_size=pool_size)(x)
            elif pool_type == 'avg':
                x = AveragePooling1D(pool_size=pool_size)(x)

        flattened = Flatten()(x)
        branches.append(flattened)
        cnn_inputs.append(input_layer)

    concatenated = Concatenate()(branches) if len(branches) > 1 else branches[0]

    dense = Dense(repr_dim, kernel_regularizer=l2(l2_reg) if l2_reg else None)(concatenated)
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
        repr_dim: int = 50,
        output_dim: int = 1,
        pds: bool = False,
        l2_reg: float = None,
        dropout_rate: float = 0.0,
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

            if dropout_rate > 0.0:
                x = Dropout(dropout_rate)(x)

            if layer % skipped_layers == 0 or skip_connection is None:
                skip_connection = x  # Update skip connection after applying activation

        flattened = Flatten()(x)
        rnn_branches.append(flattened)
        gru_inputs.append(input_layer)

    concatenated = Concatenate()(rnn_branches) if len(rnn_branches) > 1 else rnn_branches[0]

    dense = Dense(repr_dim, kernel_regularizer=l2(l2_reg) if l2_reg else None)(concatenated)
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
        repr_dim: int = 50,
        output_dim: int = 1,
        pds: bool = False,
        l2_reg: float = None,
        dropout_rate: float = 0.0,
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

            if dropout_rate > 0.0:
                x = Dropout(dropout_rate)(x)

            if layer % skipped_layers == 0:
                skip_connection = x  # Update skip connection to current x

        flattened = Flatten()(x)
        rnn_branches.append(flattened)
        gru_inputs.append(input_layer)

    concatenated = Concatenate()(rnn_branches) if len(rnn_branches) > 1 else rnn_branches[0]

    # Adjust the Dense layer's input size if necessary due to concatenation
    dense = Dense(repr_dim, kernel_regularizer=l2(l2_reg) if l2_reg else None)(concatenated)
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
        input_dims: list = None,
        gru_units: int = 64,
        gru_layers: int = 2,
        repr_dim: int = 50,
        output_dim: int = 1,
        pds: bool = False,
        l2_reg: float = None,
        dropout_rate: float = 0.0,
        activation=None,
        norm: str = None,
        name: str = 'gru'
) -> Model:
    """
    Create a model with multiple RNN (GRU) branches, each processing a different input dimension.
    The outputs of these branches are concatenated before being passed to dense layers.

    Parameters:
    - input_dims (list): List of input dimensions, one for each RNN branch. Default is [25, 25, 25].
    - gru_units (int): The number of units in each GRU layer. Default is 64.
    - repr_dim (int): The number of units in the fully connected layer. Default is 50.
    - output_dim (int): The dimension of the output layer. Default is 1 for regression tasks.
    - rnn_layers (int): The number of RNN layers in each branch. Default is 2.
    - pds (bool): If True, the model will be use PDS and there will have its representations normalized.
    - l2_reg (float): L2 regularization factor. Default is None (no regularization).
    - dropout_rate (float): The fraction of the input units to drop. Default is 0.0 (no dropout).
    - activation: Optional activation function to use. If None, defaults to LeakyReLU.
    - norm (str): Optional normalization type to use ('batch_norm' or 'layer_norm'). Default is None (no normalization).

    Note: inputs for RNNs is sequence: A 3D tensor, with shape [batch, timesteps, feature].
    In this context, features is like channels, number of variables per timesteps
    Returns:
    - Model: A Keras model instance.
    """

    if input_dims is None:
        input_dims = [25, 25, 25]

    dim_counts = Counter(input_dims)
    rnn_branches = []
    gru_inputs = []

    for dim, count in dim_counts.items():
        input_layer = Input(shape=(None, count), name=f'input_{dim}x{count}')
        x = input_layer

        for layer in range(gru_layers):
            x = GRU(units=gru_units,
                    return_sequences=True if layer < gru_layers - 1 else False,
                    kernel_regularizer=l2(l2_reg) if l2_reg else None)(x)

            if norm == 'batch_norm':
                x = BatchNormalization()(x)
            elif norm == 'layer_norm':
                x = LayerNormalization()(x)

            x = activation(x) if callable(activation) else LeakyReLU()(x)

            if dropout_rate > 0.0:
                x = Dropout(dropout_rate)(x)

        flattened = Flatten()(x)
        rnn_branches.append(flattened)
        gru_inputs.append(input_layer)

    concatenated = Concatenate()(rnn_branches) if len(rnn_branches) > 1 else rnn_branches[0]

    dense = Dense(repr_dim, kernel_regularizer=l2(l2_reg) if l2_reg else None)(concatenated)
    final_repr_output = activation(dense) if callable(activation) else LeakyReLU()(dense)

    if pds:
        # Assuming NormalizeLayer is defined elsewhere
        normalized_repr_layer = NormalizeLayer(name='normalize_layer')(final_repr_output)
        final_repr_output = normalized_repr_layer

    if output_dim > 0:
        output_layer = Dense(output_dim, name='forecast_head')(final_repr_output)
        model_output = [final_repr_output, output_layer]
    else:
        model_output = final_repr_output

    model = Model(inputs=gru_inputs, outputs=model_output, name=name)
    return model


def create_mlp(
        input_dim: int = 25,
        output_dim: int = 1,
        hiddens=None,
        skipped_layers: int = 2,
        repr_dim: int = 9,
        pds: bool = False,
        l2_reg: float = None,
        dropout_rate: float = 0.0,
        activation=None,
        norm: str = None,
        residual: bool = False,
        sam_rho: float = 0.05,
        name: str = 'mlp'
) -> Model:
    """
    Create an MLP model with fully connected dense layers, optional dropout, and configurable activation functions,
    with the option to include residual connections, batch normalization, or layer normalization.

    Parameters:
   - input_dim (int): The number of features in the input data.
    - output_dim (int): The dimension of the output layer. Default is 1 for regression tasks.
    - hiddens (list): A list of integers where each integer is the number of units in a hidden layer.
    - repr_dim (int): The number of features in the final representation vector.
    - pds (bool): If True, the model will use PDS and there will have its representations normalized.
    - l2_reg (float): L2 regularization factor. Default is None (no regularization).
    - dropout_rate (float): The fraction of the input units to drop. Default is 0.0 (no dropout).
    - activation: Optional activation function to use. If None, defaults to LeakyReLU.
    - norm (str): Optional normalization type to use ('batch_norm' or 'layer_norm'). Default is None (no normalization).
    - skipped_layers (int): Number of layers between residual connections.
    - residual (bool): If True, add residual connections for every 'skipped_layers' hidden layers.
    - sam_rho (float): Size of the neighborhood for perturbation in SAM. Default is 0.9. if 0.0, SAM is not used.

    Returns:
    - Model: A Keras model instance.
    """

    if hiddens is None:
        hiddens = [50, 50]  # Default hidden layers configuration

    if activation is None:
        activation = LeakyReLU()

    input_layer = Input(shape=(input_dim,))
    x = input_layer
    residual_layer = None

    for i, units in enumerate(hiddens):
        if i % skipped_layers == 0 and i > 0 and residual:
            if residual_layer is not None:
                # Check if projection is needed
                if x.shape[-1] != residual_layer.shape[-1]:
                    # Correct projection to match 'x' dimensions
                    residual_layer = Dense(x.shape[-1], kernel_regularizer=l2(l2_reg) if l2_reg else None,
                                           use_bias=False)(residual_layer)
                x = Add()([x, residual_layer])
            residual_layer = x  # Update the starting point for the next residual connection
        else:
            if i % skipped_layers == 0 or residual_layer is None:
                residual_layer = x

        x = Dense(units, kernel_regularizer=l2(l2_reg) if l2_reg else None)(x)

        if norm == 'batch_norm':
            x = BatchNormalization()(x)
        elif norm == 'layer_norm':
            x = LayerNormalization()(x)

        if callable(activation):
            x = activation(x)
        else:
            x = LeakyReLU()(x)

        if dropout_rate > 0.0:
            x = Dropout(dropout_rate)(x)

    # Final representation layer
    dense = Dense(repr_dim)(x)
    if pds:
        # Assuming NormalizeLayer is defined elsewhere
        repr_layer = activation(dense) if callable(activation) else LeakyReLU()(dense)
        normalized_repr_layer = NormalizeLayer(name='normalize_layer')(repr_layer)  # Custom normalization layer for PDS
        final_repr_output = normalized_repr_layer
    else:
        final_repr_output = activation(dense) if callable(activation) else LeakyReLU(name='repr_layer')(dense)

    if output_dim > 0:
        output_layer = Dense(output_dim, name='forecast_head')(final_repr_output)
        model_output = [final_repr_output, output_layer]
    else:
        model_output = final_repr_output

    if sam_rho > 0.0:
        model = SAMModel(inputs=input_layer, outputs=model_output, rho=sam_rho, name=name)
    else:
        model = Model(inputs=input_layer, outputs=model_output, name=name)
    return model


def create_mlp_moe(
        input_dim: int = 25,
        output_dim: int = 1,
        hiddens=None,
        skipped_layers: int = 2,
        repr_dim: int = 9,
        pds: bool = False,
        l2_reg: float = None,
        dropout_rate: float = 0.0,
        activation=None,
        norm: str = None,
        residual: bool = False,
        name: str = 'mlp_moe',
        expert_high_path: str = None,
        expert_low_path: str = None,
        router_hiddens=None,
        freeze_experts: bool = False,
        temperature: float = 1.0
) -> Model:
    """
    Create an MLP model with fully connected dense layers, optional dropout, and configurable activation functions,
    with the option to include residual connections, batch normalization, or layer normalization.
    Also includes a mixture of experts with pre-trained weights and a configurable gating network.

    Parameters:
    - input_dim (int): The number of features in the input data.
    - output_dim (int): The dimension of the output layer. Default is 1 for regression tasks.
    - hiddens (list): A list of integers where each integer is the number of units in a hidden layer.
    - repr_dim (int): The number of features in the final representation vector.
    - pds (bool): If True, the model will use PDS and there will have its representations normalized.
    - l2_reg (float): L2 regularization factor. Default is None (no regularization).
    - dropout_rate (float): The fraction of the input units to drop. Default is 0.0 (no dropout).
    - activation: Optional activation function to use. If None, defaults to LeakyReLU.
    - norm (str): Optional normalization type to use ('batch_norm' or 'layer_norm'). Default is None (no normalization).
    - skipped_layers (int): Number of layers between residual connections.
    - residual (bool): If True, add residual connections for every 'skipped_layers' hidden layers.
    - expert_high_path (str): Path to the stored weights of the high expert.
    - expert_low_path (str): Path to the stored weights of the low expert.
    - router_hiddens (list): A list of integers where each integer is the number of units in a hidden layer for the router.
    - freeze_experts (bool): If True, freeze the expert layers.
    - temperature (float): Temperature parameter for the softmax function in the gating network.

    Returns:
    - Model: A Keras model instance.
    """

    if hiddens is None:
        hiddens = [50, 50]  # Default hidden layers configuration

    if activation is None:
        activation = LeakyReLU()

    input_layer = Input(shape=(input_dim,))

    # Create experts using the create_mlp function
    expert_high = create_mlp(
        input_dim=input_dim,
        output_dim=1,  # Single output for regression
        hiddens=hiddens,
        skipped_layers=skipped_layers,
        repr_dim=repr_dim,
        pds=pds,
        l2_reg=l2_reg,
        dropout_rate=dropout_rate,
        activation=activation,
        norm=norm,
        residual=residual,
        name='expert_high'
    )

    expert_low = create_mlp(
        input_dim=input_dim,
        output_dim=1,  # Single output for regression
        hiddens=hiddens,
        skipped_layers=skipped_layers,
        repr_dim=repr_dim,
        pds=pds,
        l2_reg=l2_reg,
        dropout_rate=dropout_rate,
        activation=activation,
        norm=norm,
        residual=residual,
        name='expert_low'
    )

    # Load weights for experts if paths are provided
    if expert_high_path:
        expert_high.load_weights(expert_high_path)
        if freeze_experts:
            for layer in expert_high.layers:
                layer.trainable = False

    if expert_low_path:
        expert_low.load_weights(expert_low_path)
        if freeze_experts:
            for layer in expert_low.layers:
                layer.trainable = False

    final_repr_output1 = expert_high(input_layer)
    final_repr_output2 = expert_low(input_layer)

    # Gating network
    if router_hiddens is None:
        router_hiddens = hiddens
    x_router = input_layer
    for units in router_hiddens:
        x_router = Dense(units, activation=activation)(x_router)

    # Apply temperature to the softmax
    def softmax_with_temperature(logits, temperature=1.0):
        return Softmax()(logits / temperature)

    gating_logits = Dense(2)(x_router)
    gating_network = Lambda(lambda x: softmax_with_temperature(x, temperature))(gating_logits)

    # Extract weights for the experts
    expert1_weight = Lambda(lambda x: x[:, 0:1])(gating_network)
    expert2_weight = Lambda(lambda x: x[:, 1:2])(gating_network)

    # Combine experts' outputs using the gating network's weights
    expert1_weighted = Multiply()([final_repr_output1[1], expert1_weight])
    expert2_weighted = Multiply()([final_repr_output2[1], expert2_weight])
    combined_output = Add()([expert1_weighted, expert2_weighted])

    if output_dim > 0:
        output_layer = Dense(output_dim, name='forecast_head')(combined_output)
        model_output = [combined_output, output_layer]
    else:
        model_output = combined_output

    model = Model(inputs=input_layer, outputs=model_output, name=name)
    return model


def create_hybrid_model(
        tsf_extractor: Model,
        mlp_input_dim: int = 23,
        mlp_hiddens=None,
        mlp_repr_dim: int = 9,
        final_hiddens=None,
        repr_dim: int = 10,
        output_dim: int = 1,
        pds: bool = False,
        l2_reg: float = None,
        dropout_rate: float = 0.0,
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
        - repr_dim (int): The number of features in the final representation vector.
        - mlp_repr_dim (int): The number of features in the representation vector after the MLP branch.
        - mlp_hiddens (List[int]): List of integers for the MLP hidden layers.
        - final_hiddens (List[int]): List of integers for the hidden layers after concatenation.
        - pds (bool): If True, the model will be use PDS and there will have its representations normalized.
        - l2_reg (float): L2 regularization factor. Default is None (no regularization).
        - dropout_rate (float): The fraction of the input units to drop. Default is 0.0 (no dropout).
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

        if dropout_rate > 0.0:
            x_mlp = Dropout(dropout_rate)(x_mlp)

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

        if dropout_rate > 0.0:
            x_combined = Dropout(dropout_rate)(x_combined)

    # Final representation layer
    final_repr = Dense(repr_dim, kernel_regularizer=l2(l2_reg) if l2_reg else None)(x_combined)
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes data from a single file.

    Parameters:
        - file_path (str): Path to the file.
        - apply_log (bool): Whether to apply a logarithmic transformation before normalization.
        - inputs_to_use (Optional[List[str]]): List of input types to include in the dataset.
        - add_slope (bool): If True, adds slope features to the dataset.
        - outputs_to_use (Optional[List[str]]): List of output types to include in the dataset. default is both ['p'] and ['delta_p'].
        - cme_speed_threshold (float): The threshold for CME speed. CMEs with speeds below (<) this threshold will be excluded. -1
        for no cmes

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Processed input data (X) and target data (y) as numpy arrays.
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

    # Normalize targets between 0 and 1
    target_data = data[target_column]

    if cme_speed_threshold > -1:
        # Process and append CME features
        cme_features = preprocess_cme_features(data, inputs_to_use)
        combined_input = pd.concat([input_data_normalized, cme_features], axis=1)
        X = combined_input.values
    else:
        X = input_data_normalized.values.reshape((input_data_normalized.shape[0], -1, 1))

    # print shape of X along with the file name
    # print(f'X.shape: {X.shape} for {file_path}')

    y = target_data.values

    # print shape of y along with the file name
    # print(f'y.shape: {y.shape} for {file_path}')

    # Return processed X and y
    return X, y


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
) -> Tuple[np.ndarray, np.ndarray]:
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
    all_inputs, all_targets = [], []

    for file_name in os.listdir(directory_path):
        if file_name.endswith('_ie_trim.csv'):
            file_path = os.path.join(directory_path, file_name)
            X, y = load_file_data(
                file_path,
                apply_log,
                inputs_to_use,
                add_slope,
                outputs_to_use,
                cme_speed_threshold)
            all_inputs.append(X)
            all_targets.append(y)

    X_combined = np.vstack(all_inputs)
    y_combined = np.concatenate(all_targets)

    if shuffle_data:
        X_combined, y_combined = shuffle(X_combined, y_combined, random_state=seed_value)

    return X_combined, y_combined


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
        use_dict: bool = False) -> [float, str]:
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
        # Reshape X accordingly
        # The '-1' in reshape indicates that the number of samples is automatically determined
        # 'num_features' is the actual number of features in X
        # '1' is for the third dimension, typically used for weights expecting 3D input (like CNNs)
        X_reshaped = X.reshape(-1, n_features, 1)

    if add_slope:
        # n_features_list = [25] * len(inputs_to_use) * 2
        n_features_list = [25] * len(inputs_to_use) + [24] * len(inputs_to_use)
    else:
        n_features_list = [25] * len(inputs_to_use)

    X_reshaped = reshape_X(X_reshaped, n_features_list, inputs_to_use, add_slope, model.name)

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
) -> [float, str]:
    """
    Plots theactual delta (x) vs predicted delta (y) with a diagonal dotted line indicating perfect prediction.
    Different colors are used for different events

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the SEP event cme_files with normalized values.
    - cme_start_times (List[pd.Timestamp]): A list of CME start times for vertical markers.
    - event_id (str): The event ID to be displayed in the plot title.
    - model (tf.keras.Model): The trained model to be evaluated.
    - input_columns (List[str]): The list of input columns for the model.
    - cme_columns (List[str]): The list of CME feature columns.
    - using_cme (bool): Whether to use CME features. Default is False.
    - title (str): The title of the plot. Default is None.
    - inputs_to_use (List[str]): The list of input types to use. Default is None.
    - add_slope (bool): Whether to add slope features. Default is True.
    - outputs_to_use (List[str]): The list of output types to use. Default is None.
    - use_dict (bool): Whether to use the dictionary for the model. Default is False.

    Returns:
    - Tuple[float, str]: A tuple containing the MAE loss and the plot title.
    """
    global actual_changes, predicted_changes

    if add_slope:
        n_features = len(inputs_to_use) * (25 + 24)
        # n_features = len(inputs_to_use) * 25 * 2
    else:
        n_features = len(inputs_to_use) * 25

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

    if using_cme:
        # process cme features
        cme_features = preprocess_cme_features(df, inputs_to_use)
        X_reshaped = np.concatenate([X, cme_features.values], axis=1)
    else:
        # Reshape X accordingly
        # The '-1' in reshape indicates that the number of samples is automatically determined
        # 'num_features' is the actual number of features in X
        # '1' is for the third dimension, typically used for weights expecting 3D input (like CNNs)
        X_reshaped = X.reshape(-1, n_features, 1)

    if add_slope:
        # n_features_list = [25] * len(inputs_to_use) * 2
        n_features_list = [25] * len(inputs_to_use) + [24] * len(inputs_to_use)
    else:
        n_features_list = [25] * len(inputs_to_use)

    X_reshaped = reshape_X(X_reshaped, n_features_list, inputs_to_use, add_slope, model.name)

    # Evaluate the model
    if use_dict:
        res = model.predict(X_reshaped)
        predictions = res['output']
    else:
        _, predictions = model.predict(X_reshaped)

    predictions = process_predictions(predictions)

    print("Using target change approach")
    actual_changes = df['delta_log_Intensity'].values
    predicted_changes = predictions

    #     print type of actual_changes and predicted_changes and their shapes
    # print(f"Type of actual_changes: {type(actual_changes)}")
    # print(f"Type of predicted_changes: {type(predicted_changes)}")
    # print(f"Shape of actual_changes: {actual_changes.shape}")
    # print(f"Shape of predicted_changes: {predicted_changes.shape}")

    return actual_changes, predicted_changes


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
    - inputs_to_use (List[str]): List of input types to include in the dataset. Default is ['e0.5', 'e1.8', 'p'].
    - add_slope (bool): If True, adds slope features to the dataset.
    - outputs_to_use (List[str]): List of output types to include in the dataset. Default is ['p'].
    - cme_speed_threshold (float): The threshold for CME speed. CMEs with speeds below this threshold will be excluded.
    - show_avsp (bool): Whether to show the Actual vs Predicted delta plot. Default is False.
    - show_error_hist (bool): Whether to show the error histogram. Default is True.
    - prefix (str): The prefix to use for the plot file names. Default is 'testing'.
    - use_dict (bool): Whether to use the dictionary for the model. Default is False.

    Returns:
    - str: The name of the plot file.

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
    # additional_columns = ['Timestamp', 'cme_donki_time']

    plot_names = []

    cme_columns_to_zero_out = [
        'CME_DONKI_latitude', 'CME_DONKI_longitude', 'CME_DONKI_speed', 'CME_CDAW_MPA',
        'CME_CDAW_LinearSpeed', 'VlogV', 'DONKI_half_width', 'Accelaration',
        '2nd_order_speed_final', '2nd_order_speed_20R', 'CPA', 'Halo', 'Type2_Viz_Area',
        'solar_wind_speed', 'diffusive_shock', 'half_richardson_value'
    ]

    # Initialize a list to hold data for plotting
    avsp_data = []

    # Iterate over files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('_ie_trim.csv'):
            try:
                file_path = os.path.join(directory, file_name)

                # Read the SEP event cme_files
                df = read_sep_data(file_path)

                # Skip files where proton intensity is -9999
                if (df[target_column] == -9999).any():
                    continue

                df = zero_out_cme_below_threshold(df, cme_speed_threshold, cme_columns_to_zero_out)

                # # Apply time offset to align the proton and electron intensities
                # offset = pd.Timedelta(minutes=30)
                # df = apply_time_offset(df, offset)
                # Extract CME start times
                cme_start_times = extract_cme_start_times(df)
                # Extract event ID from filename
                event_id = file_name.split('_')[2]

                # Select only the input columns for the model
                # model_inputs = df[input_columns]

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
                    actual_ch, predicted_ch = plot_avsp_delta(
                        df, model, input_columns, using_cme=using_cme,
                        inputs_to_use=inputs_to_use, add_slope=add_slope,
                        outputs_to_use=outputs_to_use, use_dict=use_dict)

                    avsp_data.append((event_id, actual_ch, predicted_ch))
            except Exception as e:
                print(f"Error processing file: {file_name}")
                print(e)
                traceback.print_exc()
                continue

    if show_avsp and avsp_data:
        plot_file_location = plot_actual_vs_predicted(avsp_data, title, prefix)
        print(f"Saved plot to: {plot_file_location}")
        plot_names.append(plot_file_location)

        if show_error_hist:
            histogram_path = plot_error_dist(avsp_data, f"{title} Error Distribution", prefix)
            print(f"Saved histogram plot to: {histogram_path}")
            plot_names.append(histogram_path)

        if show_error_concentration:
            concentration_path = plot_error_concentration(avsp_data, f"{title} Error Concentration", prefix)
            print(f"Saved concentration plot to: {concentration_path}")
            plot_names.append(concentration_path)

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
        X_test: np.ndarray or List[np.ndarray],
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
        X_test: np.ndarray or List[np.ndarray],
        y_test: np.ndarray,
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
    else:
        filtered_predictions = predictions
        filtered_y_test = y_test

    # Calculate PCC
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
        use_ch: bool = True) -> Tuple:
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
            return (concatenate_seq_inputs,)
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


def asymmetric_weight_sigmoid(y_true: tf.Tensor, y_pred: tf.Tensor, eta: float = 0.5, zeta: float = 2) -> tf.Tensor:
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


def mse_pcc(
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

    # # print shape and elements of y_true and y_pred
    # print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    # print(f"y_true elements: {y_true}, y_pred elements: {y_pred}")

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

    # Compute the Mean Squared Error (MSE)
    # mse = tf.reduce_mean(mse_weights * tf.square(y_pred - y_true))

    # Compute the Mean Squared Error (MSE) with asymmetric weights
    mse = tf.reduce_mean(asym_weights * mse_weights * tf.square(y_pred - y_true))

    # Compute the Pearson Correlation Coefficient (PCC)
    y_true_centered = y_true - tf.reduce_mean(y_true)
    y_pred_centered = y_pred - tf.reduce_mean(y_pred)

    cov = tf.reduce_sum(pcc_weights * y_true_centered * y_pred_centered)
    std_y_true = tf.sqrt(tf.reduce_sum(pcc_weights * tf.square(y_true_centered)))
    std_y_pred = tf.sqrt(tf.reduce_sum(pcc_weights * tf.square(y_pred_centered)))

    pcc = cov / (std_y_true * std_y_pred + K.epsilon())

    # Combine the weighted MSE and weighted PCC with lambda_factor
    loss = mse + lambda_factor * (1.0 - pcc)

    # Return the final loss as a single scalar value
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

    # Calculate centered values
    y_true_centered = y_true - tf.reduce_mean(y_true)
    y_pred_centered = y_pred - tf.reduce_mean(y_pred)

    # Calculate covariance and standard deviations
    cov = tf.reduce_sum(weights * y_true_centered * y_pred_centered)
    std_y_true = tf.sqrt(tf.reduce_sum(weights * tf.square(y_true_centered)))
    std_y_pred = tf.sqrt(tf.reduce_sum(weights * tf.square(y_pred_centered)))

    # Calculate PCC and the final loss
    pcc = cov / (std_y_true * std_y_pred + K.epsilon())
    loss = 1.0 - pcc

    return loss


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
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


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


class IsTrainingCallback(tf.keras.callbacks.Callback):
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
