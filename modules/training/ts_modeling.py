import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
import traceback
from collections import Counter
from matplotlib.lines import Line2D
from numpy import ndarray
from scipy.signal import correlate, correlation_lags
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
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
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from typing import Tuple, List, Optional, Union

from modules.training.cme_modeling import NormalizeLayer

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
        hiddens: list[tuple],
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


# def create_mlp(
#         input_dim: int = 25,
#         output_dim: int = 1,
#         hiddens=None,
#         repr_dim: int = 9,
#         pds: bool = False,
#         l2_reg: float = None,
#         dropout_rate: float = 0.0,
#         activation=None,
#         norm: str = None,
#         name: str = 'mlp'
# ) -> Model:
#     """
#     Create an MLP model with fully connected dense layers, optional dropout, and configurable activation functions,
#     with the option to include batch normalization or layer normalization.
#
#     Parameters:
#     - input_dim (int): The number of features in the input data.
#     - output_dim (int): The dimension of the output layer. Default is 1 for regression tasks.
#     - hiddens (list): A list of integers where each integer is the number of units in a hidden layer.
#     - repr_dim (int): The number of features in the final representation vector.
#     - pds (bool): If True, the model will use PDS and there will have its representations normalized.
#     - l2_reg (float): L2 regularization factor. Default is None (no regularization).
#     - dropout_rate (float): The fraction of the input units to drop. Default is 0.0 (no dropout).
#     - activation: Optional activation function to use. If None, defaults to LeakyReLU.
#     - norm (str): Optional normalization type to use ('batch_norm' or 'layer_norm'). Default is None (no normalization).
#
#     Returns:
#     - Model: A Keras model instance.
#     """
#
#     if hiddens is None:
#         hiddens = [50]
#
#     if activation is None:
#         activation = LeakyReLU()
#
#     input_layer = Input(shape=(input_dim,))
#     x = input_layer
#
#     for units in hiddens:
#         x = Dense(units, kernel_regularizer=l2(l2_reg) if l2_reg else None)(x)
#
#         if norm == 'batch_norm':
#             x = BatchNormalization()(x)
#         elif norm == 'layer_norm':
#             x = LayerNormalization()(x)
#
#         if callable(activation):
#             x = activation(x)
#         else:
#             x = LeakyReLU()(x)
#
#         if dropout_rate > 0.0:
#             x = Dropout(dropout_rate)(x)
#
#     dense = Dense(repr_dim)(x)
#     if pds:
#         # Assuming NormalizeLayer is defined elsewhere
#         repr_layer = activation(dense) if callable(activation) else LeakyReLU()(dense)
#         normalized_repr_layer = NormalizeLayer(name='normalize_layer')(repr_layer)  # Custom normalization layer for PDS
#         final_repr_output = normalized_repr_layer
#     else:
#         final_repr_output = activation(dense) if callable(activation) else LeakyReLU(name='repr_layer')(dense)
#
#     if output_dim > 0:
#         output_layer = Dense(output_dim, name='forecast_head')(final_repr_output)
#         model_output = [final_repr_output, output_layer]
#     else:
#         model_output = final_repr_output
#
#     model = Model(inputs=input_layer, outputs=model_output, name=name)
#     return model


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
        inputs_to_use = ['e0.5', 'e1.8', 'p']
    if outputs_to_use is None:
        outputs_to_use = ['delta_p', 'p', ]

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


def build_dataset(
        directory_path: str,
        shuffle_data: bool = True,
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
        if file_name.endswith('_ie.csv'):
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
        'p': 'p_max_intensity'
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
        prefix: str = 'testing') -> [float, str]:
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

    Returns:
    - Tuple[float, str]: A tuple containing the MAE loss and the plot title.
    """
    global actual_changes, predicted_changes
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
    _, predictions = model.predict(X_reshaped)
    predictions = process_predictions(predictions)

    # if target change then we need to convert prediction into actual value
    if 'p' in inputs_to_use and 'delta_p' in outputs_to_use:
        predictions_plot = p_t_log + predictions
        if show_changes:
            actual_changes = df['delta_log_Intensity'].values - 1  # offset by 1
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
    if 'p' in inputs_to_use and 'delta_p' in outputs_to_use and show_persistent:
        plt.plot(timestamps, p_t_log, label='Persistent Model', color='black', linestyle=':', linewidth=tlw)
    if 'p' in inputs_to_use and 'delta_p' in outputs_to_use and show_changes:
        plt.scatter(timestamps, actual_changes, color='gray', label='Actual Changes', alpha=0.5, s=ssz)
        plt.scatter(timestamps, predicted_changes, color='purple', label='Predicted Changes', alpha=0.5, s=ssz)
    # Add a black horizontal line at log(0.05) on the y-axis and create a handle for the legend

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
        prefix: str = 'testing') -> List[str]:
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
    - prefix (str): The prefix to use for the plot file names. Default is 'testing'.

    Returns:
    - str: The name of the plot file.

    The function assumes that the SEP event files are named in the format 'sep_event_X_filled_ie.csv',
    where 'X' is the event ID. It skips files where the proton intensity is -9999.
    Each file will be processed to plot actual vs predicted proton intensity and electron intensity.
    A MAE score will be printed for each file.
    """

    if inputs_to_use is None:
        inputs_to_use = ['e0.5', 'e1.8', 'p']

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
        if file_name.endswith('_ie.csv'):
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
                    prefix=prefix)

                print(f"Processed file: {file_name} with MAE: {mae_loss}")
                plot_names.append(plotname)

                if show_avsp:
                    actual_ch, predicted_ch = plot_avsp_delta(
                        df, model, input_columns, using_cme=using_cme,
                        inputs_to_use=inputs_to_use, add_slope=add_slope,
                        outputs_to_use=outputs_to_use)

                    avsp_data.append((event_id, actual_ch, predicted_ch))
            except Exception as e:
                print(f"Error processing file: {file_name}")
                print(e)
                traceback.print_exc()
                continue

    if show_avsp and avsp_data:
        # Plotting logic here...
        plt.figure(figsize=(10, 7))  # Adjust size as needed
        # Flatten all actual values to find global min and max for color mapping
        # all_actual_values = np.concatenate([actual for _, actual, _ in avsp_data])
        norm = plt.Normalize(-2.5, 2.5)
        cmap = plt.cm.viridis  # Choose a colormap

        # Create a scatter plot for each set of actual vs predicted values
        for event_id, actual, predicted in avsp_data:
            plt.scatter(actual, predicted, c=actual, cmap=cmap, norm=norm, label=f'{event_id}', alpha=0.7, s=12)

        # Add a diagonal line for perfect prediction
        # min_val = min(min(actual.min(), predicted.min()) for _, actual, predicted in avsp_data)
        # max_val = max(max(actual.max(), predicted.max()) for _, actual, predicted in avsp_data)
        plt.plot([-2.5, 2.5], [-2.5, 2.5], 'k--', label='Perfect Prediction')

        plt.xlabel('Actual Changes')
        plt.ylabel('Predicted Changes')
        plt.title(f"{title}\n{prefix}_Actual_vs_Predicted_Changes")

        # Create colorbar as the legend for actual values
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label='Actual Value', extend='both')
        # plt.legend()
        plt.grid(True)

        # Save the plot
        plot_filename = f"{title}_{prefix}_actual_vs_predicted_changes.png"
        plt.savefig(plot_filename)
        plt.close()

        # Return the file location
        file_location = os.path.abspath(plot_filename)
        print(f"Saved plot to: {file_location}")
        plot_names.append(file_location)

    return plot_names


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


def evaluate_model(
        model: tf.keras.Model,
        X_test: np.ndarray or List[np.ndarray],
        y_test: np.ndarray) -> float:
    """
    Evaluates a given model using Mean Absolute Error (MAE) on the provided test cme_files.

    Parameters:
    - model (tf.keras.Model): The trained model to evaluate.
    - X_test (np.ndarray): Test features.
    - y_test (np.ndarray): True target values for the test set.
    - symlog1p (bool): Whether to apply inverse symlog1p transformation to predictions. Default is False.

    Returns:
    - float: The MAE loss of the model on the test cme_files.
    """
    # Make predictions
    _, predictions = model.predict(X_test)

    # Process predictions
    predictions = process_predictions(predictions)
    y_test = process_predictions(y_test)

    # Calculate MAE
    mae_loss = mean_absolute_error(y_test, predictions)

    return mae_loss


def evaluate_model_cond(
        model: tf.keras.Model,
        X_test: np.ndarray or List[np.ndarray],
        y_test: np.ndarray,
        below_threshold: float = None,
        above_threshold: float = None) -> float:
    """
    Evaluates a given model using Mean Absolute Error (MAE) on the provided test data,
    with an option to conditionally calculate MAE based on specified thresholds.

    Parameters:
    - model (tf.keras.Model): The trained model to evaluate.
    - X_test (np.ndarray): Test features.
    - y_test (np.ndarray): True target values for the test set.
    - below_threshold (float, optional): The lower bound threshold for y_test to be included in MAE calculation.
    - above_threshold (float, optional): The upper bound threshold for y_test to be included in MAE calculation.

    Returns:
    - float: The MAE loss of the model on the filtered test data.
    """
    # Make predictions
    _, predictions = model.predict(X_test)

    # Process predictions
    predictions = process_predictions(predictions)
    y_test = process_predictions(y_test)

    # Filter y_test and predictions based on thresholds
    if below_threshold is not None and above_threshold is not None:
        mask = (y_test >= above_threshold) | (y_test <= below_threshold)
    elif below_threshold is not None:
        mask = y_test <= below_threshold
    elif above_threshold is not None:
        mask = y_test >= above_threshold
    else:
        mask = np.ones_like(y_test, dtype=bool)

    filtered_predictions = predictions[mask]
    filtered_y_test = y_test[mask]

    # Calculate MAE
    mae_loss = mean_absolute_error(filtered_y_test, filtered_predictions)

    return mae_loss


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
        return prepare_hybrid_inputs(X,
                                     tsf_input_dims=n_features_list,
                                     with_slope=add_slope,
                                     mlp_input_dim=20 + len(inputs_to_use))
    else:
        raise ValueError("Invalid model_type. Must be 'mlp', 'cnn' or 'rnn'.")


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


def get_loss(loss_key: str = 'mse'):
    """
    given the key, return the appropiate loss function for the model

    :param loss_key: key for the loss function
    :return: loss string or function for Tensorflow compile loss
    """

    if loss_key == 'mse':
        return 'mse'
    elif loss_key == 'exp_mse':
        def exp_mse(y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)
            return tf.exp(mse) - 1

        return exp_mse
    elif loss_key == 'var_mse':
        def var_mse(y_true, y_pred):
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            variance_loss = -tf.reduce_mean(tf.square(y_pred - tf.reduce_mean(y_pred)))
            total_loss = mse_loss + 0.08 * variance_loss  # Adjust the weighting factor as needed
            return total_loss

        return var_mse


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
