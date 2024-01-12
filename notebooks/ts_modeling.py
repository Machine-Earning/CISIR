import random
import traceback
from typing import Tuple, List, Any
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, LeakyReLU, Concatenate, GRU
from tensorflow.keras.models import Model
import os

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


def create_cnns(
        input_dims: list = None,
        filters: int = 32,
        kernel_size: int = 10,
        conv_layers: int = 2,
        repr_dim: int = 50,
        output_dim: int = 1
) -> Model:
    """
    Create a model with multiple CNN branches, each processing a different input dimension.
    The outputs of these branches are concatenated before being passed to dense layers.

    Parameters:
    - input_dims (list): List of input dimensions, one for each CNN branch. Default is [25, 25, 25].
    - output_dim (int): The dimension of the output layer. Default is 1 for regression tasks.
    - filters (int): The number of filters in each convolutional layer. Default is 32.
    - kernel_size (int): The size of the kernel in the convolutional layers. Default is 10.
    - repr_dim (int): The number of units in the fully connected layer. Default is 50.
    - conv_layers (int): The number of convolutional layers in each branch. Default is 2.

    Returns:
    - Model: A Keras model instance.
    """

    if input_dims is None:
        input_dims = [25, 25, 25]

    cnn_branches = []
    cnn_inputs = []

    for i, dim in enumerate(input_dims):
        # Define the input layer for this branch
        input_layer = Input(shape=(dim, 1), name=f'input_{i}')
        x = input_layer

        # Add convolutional layers with LeakyReLU activation
        for _ in range(conv_layers):
            x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
            x = LeakyReLU()(x)

        # Flatten the output for concatenation
        flattened = Flatten()(x)

        cnn_branches.append(flattened)
        cnn_inputs.append(input_layer)

    # Concatenate the outputs of all branches
    concatenated = Concatenate()(cnn_branches) if len(cnn_branches) > 1 else cnn_branches[0]

    # Add a fully connected layer with LeakyReLU activation
    dense = Dense(repr_dim)(concatenated)
    repr_output = LeakyReLU(name='repr_layer')(dense)

    # Determine the model's output based on output_dim
    if output_dim > 0:
        # Output layer for regression or classification tasks
        output_layer = Dense(output_dim, name='forecast_head')(repr_output)
        model_output = [repr_output, output_layer]
    else:
        # Only output the representation layer
        model_output = repr_output

    # Create the model
    model = Model(inputs=cnn_inputs, outputs=model_output)

    return model


def create_rnns(
        input_dims: list = None,
        gru_units: int = 64,
        rnn_layers: int = 2,
        repr_dim: int = 50,
        output_dim: int = 1) -> Model:
    """
    Create a model with multiple RNN (GRU) branches, each processing a different input dimension.
    The outputs of these branches are concatenated before being passed to dense layers.

    Parameters:
    - input_dims (list): List of input dimensions, one for each RNN branch. Default is [25, 25, 25].
    - gru_units (int): The number of units in each GRU layer. Default is 64.
    - repr_dim (int): The number of units in the fully connected layer. Default is 50.
    - output_dim (int): The dimension of the output layer. Default is 1 for regression tasks.
    - rnn_layers (int): The number of RNN layers in each branch. Default is 2.

    Returns:
    - Model: A Keras model instance.
    """

    if input_dims is None:
        input_dims = [25, 25, 25]

    rnn_branches = []
    rnn_inputs = []

    for i, dim in enumerate(input_dims):
        # Define the input layer for this branch
        input_layer = Input(shape=(dim, 1), name=f'input_{i}')
        x = input_layer

        # Add GRU layers with LeakyReLU activation
        for _ in range(rnn_layers):
            x = GRU(units=gru_units, return_sequences=True if _ < rnn_layers - 1 else False)(x)
            x = LeakyReLU()(x)

        # Flatten the output for concatenation
        flattened = Flatten()(x)

        rnn_branches.append(flattened)
        rnn_inputs.append(input_layer)

    # Concatenate the outputs of all branches
    concatenated = Concatenate()(rnn_branches) if len(rnn_branches) > 1 else rnn_branches[0]

    # Add a fully connected layer with LeakyReLU activation
    dense = Dense(repr_dim)(concatenated)
    repr_output = LeakyReLU(name='repr_layer')(dense)

    # Determine the model's output based on output_dim
    if output_dim > 0:
        # Output layer for regression or classification tasks
        output_layer = Dense(output_dim, name='forecast_head')(repr_output)
        model_output = [repr_output, output_layer]
    else:
        # Only output the representation layer
        model_output = repr_output

    # Create the model
    model = Model(inputs=rnn_inputs, outputs=model_output)

    return model


def create_mlp(input_dim: int = 25, output_dim: int = 1, hiddens=None, repr_dim: int = 9) -> Model:
    """
    Create an MLP model with fully connected dense layers.

    Parameters:
    - input_dim (int): The number of features in the input data.
    - output_dim (int): The dimension of the output layer. Default is 1 for regression tasks.
    - hiddens (list): A list of integers where each integer is the number of units in a hidden layer.
    - repr_dim (int): The number of features in the final representation vector.

    Returns:
    - Model: A Keras model instance.
    """

    # Define the input layer
    if hiddens is None:
        hiddens = [50]
    input_layer = Input(shape=(input_dim,))

    # Create hidden layers with LeakyReLU activation
    x = input_layer
    for units in hiddens:
        x = Dense(units)(x)
        x = LeakyReLU()(x)

    # Final representation layer
    repr_output = Dense(repr_dim)(x)
    repr_output = LeakyReLU(name='repr_layer')(repr_output)

    # Output layer
    output_layer = Dense(output_dim, name='forecast_head')(repr_output)

    # Create the model
    model = Model(inputs=input_layer, outputs=[repr_output, output_layer])

    return model


# def create_fork_model(cnn_input_dims=None, mlp_input_dim: int = 23, output_dim: int = 1, repr_dim: int = 10,
#                       cnn_filters: int = 32, cnn_kernel_size: int = 10, cnn_repr_dim: int = 50, mlp_repr_dim: int = 9,
#                       mlp_hiddens=None, final_hiddens=None) -> Model:
#     """
#     Create a fork-like neural network model with multiple CNN branches and an MLP branch.
#
#     Parameters:
#     - cnn_input_dims (list[int]): List of input dimensions for each CNN branch.
#     - mlp_input_dim (int): The number of features for the MLP branch.
#     - output_dim (int): The dimension of the output layer.
#     - repr_dim (int): The number of features in the final representation vector.
#     - cnn_filters (int): The number of filters in the CNN layers.
#     - cnn_kernel_size (int): The size of the kernel in the CNN layers.
#     - cnn_repr_dim (int): The number of features in the representation vector after the CNN branches.
#     - mlp_repr_dim (int): The number of features in the representation vector after the MLP branch.
#     - mlp_hiddens (List[int]): List of integers for the MLP hidden layers.
#     - final_hiddens (List[int]): List of integers for the hidden layers after concatenation.
#
#     Returns:
#     - Model: A Keras model instance.
#     """
#
#     if cnn_input_dims is None:
#         cnn_input_dims = [25, 25, 25]
#     if final_hiddens is None:
#         final_hiddens = [12]
#     if mlp_hiddens is None:
#         mlp_hiddens = [18]
#
#     cnn_branches = []
#     cnn_inputs = []
#
#     # Create CNN branches
#     for i, dim in enumerate(cnn_input_dims):
#         cnn_input = Input(shape=(dim, 1), name=f'cnn_input_{i}')
#         x_cnn = Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, padding='same')(cnn_input)
#         x_cnn = LeakyReLU()(x_cnn)
#         x_cnn = Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, padding='same')(x_cnn)
#         x_cnn = LeakyReLU()(x_cnn)
#         x_cnn = Flatten()(x_cnn)
#         x_cnn = Dense(cnn_repr_dim)(x_cnn)
#         cnn_repr = LeakyReLU()(x_cnn)
#         cnn_branches.append(cnn_repr)
#         cnn_inputs.append(cnn_input)
#
#     # MLP Branch
#     mlp_input = Input(shape=(mlp_input_dim,), name='mlp_input')
#     x_mlp = mlp_input
#     for units in mlp_hiddens:
#         x_mlp = Dense(units)(x_mlp)
#         x_mlp = LeakyReLU()(x_mlp)
#     x_mlp = Dense(mlp_repr_dim)(x_mlp)
#     mlp_repr = LeakyReLU()(x_mlp)
#
#     # Concatenate the outputs of CNN and MLP branches
#     concatenated = Concatenate()(cnn_branches + [mlp_repr])
#
#     # Additional MLP Layer(s) after concatenation
#     x_combined = concatenated
#     for units in final_hiddens:
#         x_combined = Dense(units)(x_combined)
#         x_combined = LeakyReLU()(x_combined)
#
#     # Final representation layer
#     final_repr = Dense(repr_dim)(x_combined)
#     final_repr = LeakyReLU(name='repr_layer')(final_repr)
#
#     # Final output layer
#     forecast_head = Dense(output_dim, activation='linear', name='forecast_head')(final_repr)
#
#     # Create the model
#     model = Model(inputs=cnn_inputs + [mlp_input], outputs=[final_repr, forecast_head])
#
#     return model


def create_hybrid_model(
        tsf_extractor: Model,
        mlp_input_dim: int = 23,
        mlp_hiddens=None,
        mlp_repr_dim: int = 9,
        final_hiddens=None,
        repr_dim: int = 10,
        output_dim: int = 1

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

    Returns:
    - Model: A Keras model instance.
    """

    if final_hiddens is None:
        final_hiddens = [12]
    if mlp_hiddens is None:
        mlp_hiddens = [18]

    # MLP Branch
    mlp_input = Input(shape=(mlp_input_dim,), name='mlp_input')
    x_mlp = mlp_input
    for units in mlp_hiddens:
        x_mlp = Dense(units)(x_mlp)
        x_mlp = LeakyReLU()(x_mlp)
    x_mlp = Dense(mlp_repr_dim)(x_mlp)
    mlp_repr = LeakyReLU()(x_mlp)

    # Concatenate the outputs of TSF Extractor and MLP branch
    concatenated = Concatenate()([tsf_extractor.output, mlp_repr])

    # Additional MLP Layer(s) after concatenation
    x_combined = concatenated
    for units in final_hiddens:
        x_combined = Dense(units)(x_combined)
        x_combined = LeakyReLU()(x_combined)

    # Final representation layer
    final_repr = Dense(repr_dim)(x_combined)
    final_repr = LeakyReLU(name='final_repr_layer')(final_repr)

    # Final output layer
    forecast_head = Dense(output_dim, activation='linear', name='forecast_head')(final_repr)

    # Create the model
    model = Model(inputs=[tsf_extractor.input, mlp_input], outputs=[final_repr, forecast_head])

    return model


def build_dataset(directory_path: str, shuffle_data: bool = True, apply_log: bool = True, norm_target: bool = False,
                  inputs_to_use: List[str] = None, add_slope: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads SEP event files from the specified directory, processes them to extract
    input and target data, normalizes the values between 0 and 1 for the columns
    of interest, excludes rows where proton intensity is -9999, and optionally shuffles the data.

     Parameters:
        - directory_path (str): Path to the directory containing the sep_event_X files.
        - shuffle_data (bool): If True, shuffle the data before returning.
        - apply_log (bool): Whether to apply a logarithmic transformation before normalization.
        - norm_target (bool): Whether to normalize the target data. False by default.
        - inputs_to_use (List[str]): List of input types to include in the dataset. Default is ['e0.5', 'e1.8', 'p'].
        - add_slope (bool): If True, adds slope features to the dataset.



    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing the normalized input data (X) and target data (y).
    """

    if inputs_to_use is None:
        inputs_to_use = ['e0.5', 'e1.8', 'p']

    all_inputs = []
    all_targets = []

    # Dynamically define input columns based on inputs_to_use
    input_columns = []
    for input_type in inputs_to_use:
        input_columns += [f'{input_type}_tminus{i}' for i in range(24, 0, -1)] + [f'{input_type}_t']

    target_column = 'Proton Intensity'

    # Loop through each file in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith('_ie.csv'):
            file_path = os.path.join(directory_path, file_name)
            data = pd.read_csv(file_path)

            # Exclude rows where proton intensity is -9999
            data = data[data[target_column] != -9999]

            # Apply logarithmic transformation (if specified)
            if apply_log:
                data[input_columns] = np.log(data[input_columns] + 1)  # Adding 1 to avoid log(0)
                data[target_column] = np.log(data[target_column] + 1)  # Adding 1 to avoid log(0)

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
                        # print(f"Adding {slope_column} to input data...")
                        input_data_normalized[slope_column] = slope_values[:, slope_index]
                    # input_columns.extend(slope_column_names)

            # print the columns one by one
            # for col in input_data_normalized.columns:
            #     print(col)
            # order - e0.5, e1.8, p, e0.5 slope, e1.8 slope, p slope

            # Normalize targets between 0 and 1
            target_data = data[[target_column]]
            if norm_target:
                target_data_normalized = (target_data - target_data.min()) / (target_data.max() - target_data.min())
            else:
                target_data_normalized = target_data

            # Reshape inputs to be in the format [samples, time steps, features]
            # X = input_data_normalized.values.reshape((input_data_normalized.shape[0], 75, 1))
            X = input_data_normalized.values.reshape((input_data_normalized.shape[0], -1, 1))

            # Flatten targets to 1D array
            y = target_data_normalized.values.flatten()

            # Append to list
            all_inputs.append(X)
            all_targets.append(y)

    # Combine all input and target data
    X_combined = np.vstack(all_inputs)
    y_combined = np.concatenate(all_targets)

    # Shuffle the data if the flag is True
    if shuffle_data:
        X_combined, y_combined = shuffle(X_combined, y_combined, random_state=seed_value)

    return X_combined, y_combined


def generate_slope_column_names(inputs_to_use: List[str]) -> List[str]:
    """
    Generate slope column names for each input type based on the specified time steps.

    For each input type, this function generates column names for slopes calculated between
    consecutive time steps. It includes slopes from 'tminus24' to 'tminus2', and a special case
    for the slope from 'tminus1' to 't'.

    Parameters:
    - inputs_to_use (List[str]): A list of input types for which to generate slope column names.

    Returns:
    - List[str]: A list of slope column names for all specified input types.
    """
    slope_column_names = []
    for input_type in inputs_to_use:
        # Slopes from tminus24 to tminus2
        for i in range(24, 1, -1):
            slope_column_names.append(f'{input_type}_slope_tminus{i}_to_tminus{i - 1}')
        # Slope from tminus1 to t
        slope_column_names.append(f'{input_type}_slope_tminus1_to_t')
    return slope_column_names


def compute_slope(data: pd.DataFrame, input_type: str) -> np.ndarray:
    """
    Compute the slope for a given input type across its time-series columns.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - input_type (str): The input type for which to compute the slope (e.g., 'e0.5').

    Returns:
    - np.ndarray: Array of slopes for the specified input type.
    """
    columns = [f'{input_type}_tminus{i}' for i in range(24, 0, -1)] + [f'{input_type}_t']
    input_values = data[columns].values
    slopes = np.diff(input_values, axis=1)  # /1 for 1 unit
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
    - data (pd.DataFrame or pd.Series): The pandas DataFrame or Series to be normalized.

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

    # Preallocate a dictionary to store preprocessed data
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
    - df (pd.DataFrame): The DataFrame containing the data.
    - threshold (float): The CME speed threshold.
    - cme_columns (List[str]): List of CME column names to zero out.

    Returns:
    - pd.DataFrame: The DataFrame with updated CME columns.
    """
    mask = df['CME_DONKI_speed'] < threshold
    for column in cme_columns:
        df.loc[mask, column] = 0
    return df


def build_full_dataset(
        directory_path: str,
        shuffle_data: bool = True,
        apply_log: bool = True,
        norm_target: bool = False,
        inputs_to_use: List[str] = None,
        add_slope: bool = True,
        cme_speed_threshold: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads SEP event files from the specified directory, processes them to extract
    input and target data including cme features, normalizes the values between 0 and 1 for the columns
    of interest, excludes rows where proton intensity is -9999, and optionally shuffles the data.

    Parameters:
    - directory_path (str): Path to the directory containing the sep_event_X files.
    - shuffle_data (bool): If True, shuffle the data before returning.
    - apply_log (bool): Whether to apply a logarithmic transformation before normalization.
    - norm_target (bool): Whether to normalize the target data.
    - inputs_to_use (List[str]): List of input types to include in the dataset.
    - add_slope (bool): If True, adds slope features to the dataset.
    - cme_speed_threshold (float): The threshold for CME speed. CMEs with speeds below (<) this threshold will be excluded.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing the normalized input data (X) and target data (y).
    """

    if inputs_to_use is None:
        inputs_to_use = ['e0.5', 'e1.8', 'p']

    all_inputs = []
    all_targets = []

    # Dynamically define input columns based on inputs_to_use
    input_columns = []
    for input_type in inputs_to_use:
        input_columns += [f'{input_type}_tminus{i}' for i in range(24, 0, -1)] + [f'{input_type}_t']

    target_column = 'Proton Intensity'

    cme_columns_to_zero_out = [
        'CME_DONKI_latitude', 'CME_DONKI_longitude', 'CME_DONKI_speed', 'CME_CDAW_MPA',
        'CME_CDAW_LinearSpeed', 'VlogV', 'DONKI_half_width', 'Accelaration',
        '2nd_order_speed_final', '2nd_order_speed_20R', 'CPA', 'Halo', 'Type2_Viz_Area',
        'solar_wind_speed', 'diffusive_shock', 'half_richardson_value'
    ]

    # Loop through each file in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith('_ie.csv'):
            file_path = os.path.join(directory_path, file_name)
            data = pd.read_csv(file_path)

            # Exclude rows where proton intensity is -9999
            data = data[data[target_column] != -9999]

            # Zero out CME values for rows with speed below threshold
            data = zero_out_cme_below_threshold(data, cme_speed_threshold, cme_columns_to_zero_out)

            # Apply logarithmic transformation (if specified)
            if apply_log:
                data[input_columns] = np.log1p(data[input_columns])  # Adding 1 to avoid log(0)
                data[target_column] = np.log1p(data[target_column])  # Adding 1 to avoid log(0)

            # Normalize inputs between 0 and 1
            input_data = data[input_columns]
            input_data_normalized = min_max_norm(input_data)

            # Compute and add slopes (if specified)
            if add_slope:
                for input_type in inputs_to_use:
                    # print(f"Computing slopes for {input_type}...")
                    slope_column_names = generate_slope_column_names([input_type])  # Generate for current input type
                    slope_values = compute_slope(input_data_normalized, input_type)
                    for slope_column, slope_index in zip(slope_column_names, range(slope_values.shape[1])):
                        # print(f"Adding {slope_column} to input data...")
                        input_data_normalized[slope_column] = slope_values[:, slope_index]
                    # input_columns.extend(slope_column_names)

            # print the columns one by one
            # for col in input_data_normalized.columns:
            #     print(col)
            # order - e0.5, e1.8, p, e0.5 slope, e1.8 slope, p slope

            # Normalize targets between 0 and 1
            target_data = data[[target_column]]
            if norm_target:
                target_data_normalized = min_max_norm(target_data)
            else:
                target_data_normalized = target_data

            # Process and append CME features
            cme_features = preprocess_cme_features(data, inputs_to_use)
            combined_input = pd.concat([input_data_normalized, cme_features], axis=1)

            # # Get the order of the CME features
            # cme_feature_order = cme_features.columns.tolist()
            #
            # # Now, cme_feature_order contains the names of the CME features in the order they appear
            # print("Order of CME features:", cme_feature_order)

            X = combined_input.values  # TODO: check if X needs to be reshaped

            # Flatten targets to 1D array
            y = target_data_normalized.values.flatten()

            # check_nan_in_dataset(combined_input.values, f"in file {file_name}")

            # save  combined input to a file for debuggin
            # combined_input.to_csv(f"combined_input_{file_name}.csv")

            # Append to list
            all_inputs.append(X)
            all_targets.append(y)

    # Combine all input and target data
    X_combined = np.vstack(all_inputs)
    y_combined = np.concatenate(all_targets)

    # Shuffle the data if the flag is True
    if shuffle_data:
        X_combined, y_combined = shuffle(X_combined, y_combined, random_state=seed_value)

    return X_combined, y_combined


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

    # Normalize the synthetic data
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
    data (np.ndarray): The synthetic dataset with shape (n_samples, n_timesteps, 1).
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
    Reads the SEP event data from a CSV file.

    Parameters:
    - file_path (str): The path to the SEP event CSV file.

    Returns:
    - DataFrame: The SEP event data as a pandas DataFrame.
    """
    return pd.read_csv(file_path)


def normalize_flux(df: pd.DataFrame, columns: List[str], apply_log: bool = True) -> pd.DataFrame:
    """
    Normalizes the specified flux intensity columns in the DataFrame between 0 and 1.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the SEP event data.
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
    - df (pd.DataFrame): The DataFrame containing the SEP event data.

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
        target_column: str,
        normalize_target: bool = False,
        using_cme: bool = False,
        model_type: str = "cnn",
        title: str = None,
        inputs_to_use: List[str] = None,
        add_slope: bool = True) -> [float, str]:
    """
    Plots the SEP event data with actual and predicted proton intensities, electron intensity,
    and evaluates the model's performance using MAE.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the SEP event data with normalized values.
    - cme_start_times (List[pd.Timestamp]): A list of CME start times for vertical markers.
    - event_id (str): The event ID to be displayed in the plot title.
    - model (tf.keras.Model): The trained model to be evaluated.
    - input_columns (List[str]): The list of input columns for the model.
    - cme_columns (List[str]): The list of CME feature columns.
    - target_column (str): The column name of the target variable.
    - normalize_target (bool): Whether to normalize the target column. Default is False.
    - using_cme (bool): Whether to use CME features. Default is False.
    - model_type (str): The type of model used. Default is 'cnn'.
    - title (str): The title of the plot. Default is None.
    - inputs_to_use (List[str]): The list of input types to use. Default is None.
    - add_slope (bool): Whether to add slope features. Default is True.

    Returns:
    - Tuple[float, str]: A tuple containing the MAE loss and the plot title.
    """
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

    if normalize_target:
        df_norm[target_column] = normalize_flux(df, [target_column], apply_log=True)[target_column]
        y_true = df_norm[target_column].values
    else:
        y_true = df[target_column].values
        y_true = np.log1p(y_true)  # Log transform target if normalization is not applied

    if using_cme:
        # process cme features
        cme_features = preprocess_cme_features(df, inputs_to_use)
        X_reshaped = np.concatenate([X, cme_features.values], axis=1)
    else:
        # Reshape X accordingly
        # The '-1' in reshape indicates that the number of samples is automatically determined
        # 'num_features' is the actual number of features in X
        # '1' is for the third dimension, typically used for models expecting 3D input (like CNNs)
        X_reshaped = X.reshape(-1, n_features, 1)

    if add_slope:
        n_features_list = [25] * len(inputs_to_use) + [24] * len(inputs_to_use)
    else:
        n_features_list = [25] * len(inputs_to_use)

    if model_type == "cnn":
        X_reshaped = prepare_cnn_inputs(X_reshaped, n_features_list, add_slope)
    elif model_type == "rnn":
        X_reshaped = prepare_rnn_inputs(X_reshaped, n_features_list, add_slope)
    elif model_type == "cnn-hybrid":
        X_reshaped = prepare_hybrid_inputs(X_reshaped,
                                           tsf_extractor_type='cnn',
                                           tsf_input_dims=n_features_list,
                                           with_slope=add_slope,
                                           mlp_input_dim=20 + len(inputs_to_use))
    elif model_type == "rnn-hybrid":
        X_reshaped = prepare_hybrid_inputs(X_reshaped,
                                           tsf_extractor_type='rnn',
                                           tsf_input_dims=n_features_list,
                                           with_slope=add_slope,
                                           mlp_input_dim=20 + len(inputs_to_use))

    # Evaluate the model
    _, predictions = model.predict(X_reshaped)
    mae_loss = mean_absolute_error(y_true, predictions.flatten())
    # print(f"Mean Absolute Error (MAE) on the data: {mae_loss}")

    lw = .65  # Line width for the plots
    tlw = 2.75  # Thicker line width for the actual and predicted lines

    # Plot the data
    plt.figure(figsize=(15, 10), facecolor='white')
    plt.plot(timestamps, y_true, label='Actual Proton ln(Intensity)', color='blue', linewidth=tlw)
    plt.plot(timestamps, predictions.flatten(), label='Predicted Proton ln(Intensity)', color='red', linewidth=tlw)
    # electron_line, = plt.plot(t_timestamps, df['t'], label='Electron Intensity', color='blue')
    plt.plot(t_timestamps, e05_intensity_log, label='E 0.5 ln(Intensity)', color='orange', linewidth=lw)
    if 'e1.8' in inputs_to_use:
        plt.plot(t_timestamps, e18_intensity_log, label='E 1.8 ln(Intensity)', color='yellow', linewidth=lw)
    # Add a black horizontal line at log(0.05) on the y-axis and create a handle for the legend
    # threshold_value = 0.1
    # threshold_line = plt.axhline(y=threshold_value, color='black', linestyle='--', linewidth=lw, label='Threshold')

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
    plt.title(f'{title} - SEP Event {event_id} - MAE: {mae_loss:.4f}')

    # Extract handles and labels for the plot's elements
    handles, labels = plt.gca().get_legend_handles_labels()

    # Add custom legend handles for the threshold and CME lines
    handles.extend([cme_line])
    labels.extend(["CME Start Time"])

    plt.legend(handles=handles, labels=labels)

    # Save the plot to a file with the MAE in the file name
    file_name = f'{title}_SEP_Event_{event_id}_MAE_{mae_loss:.4f}.png'
    plt.savefig(file_name)

    # plt.show()
    # Close the plot
    plt.close()

    # Return the file location
    file_location = os.path.abspath(file_name)
    print(f"Saved plot to: {file_location}")
    return mae_loss, file_name


def process_sep_events(
        directory: str,
        model: tf.keras.Model,
        using_cme: bool = False,
        model_type: str = "cnn",
        title: str = None,
        inputs_to_use: List[str] = None,
        add_slope: bool = True,
        cme_speed_threshold: float = 0
) -> List[str]:
    """
    Processes SEP event files in the specified directory, normalizes flux intensities, predicts proton intensities,
    plots the results, and calculates the MAE for each file.

    Parameters:
    - directory (str): Path to the directory containing the SEP event files.
    - model (tf.keras.Model): The trained machine learning model for predicting proton intensity.
    - using_cme (bool): Whether to use CME features. Default is False.
    - model_type (str): The type of model used. Default is "cnn".
    - title (str): The title of the plot. Default is None.
    - inputs_to_use (List[str]): List of input types to include in the dataset. Default is ['e0.5', 'e1.8', 'p'].
    - add_slope (bool): If True, adds slope features to the dataset.
     - cme_speed_threshold (float): The threshold for CME speed. CMEs with speeds below this threshold will be excluded.

    Returns:
    - str: The name of the plot file.

    The function assumes that the SEP event files are named in the format 'sep_event_X_filled_ie.csv',
    where 'X' is the event ID. It skips files where the proton intensity is -9999.
    Each file will be processed to plot actual vs predicted proton intensity and electron intensity.
    A MAE score will be printed for each file.
    """

    if inputs_to_use is None:
        inputs_to_use = ['e0.5', 'e1.8', 'p']

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

    # Iterate over files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('_ie.csv'):
            try:
                file_path = os.path.join(directory, file_name)

                # Read the SEP event data
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
                    input_columns, target_column, using_cme=using_cme,
                    model_type=model_type, title=title,
                    inputs_to_use=inputs_to_use, add_slope=add_slope)

                print(f"Processed file: {file_name} with MAE: {mae_loss}")

                plot_names.append(plotname)
            except Exception as e:
                print(f"Error processing file: {file_name}")
                print(e)
                traceback.print_exc()
                continue

    return plot_names


def plot_sample_with_cme(data: np.ndarray, cme_features_names: list = None, sample_index: int = None) -> None:
    """
    Plots a random sample from the dataset, showing both time-series and CME features.
    If a sample index is provided, plots that specific sample.

    Parameters:
    - data (np.ndarray): The dataset with shape (n_samples, n_features).
    - cme_features_names (list): A list of names for the CME features.
    - sample_index (int, optional): The index of the specific sample to plot.
                                   If None, a random sample will be plotted. Defaults to None.
    """

    if sample_index is None:
        sample_index = np.random.randint(low=0, high=data.shape[0])

    if cme_features_names is None:
        cme_features_names = ['VlogV_norm', 'CME_DONKI_speed_norm', 'Linear_Speed_norm', '2nd_order_speed_final_norm',
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

    # Plot the time-series data
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
    - data (np.ndarray): The dataset with shape (n_samples, n_features).
    - cme_start_index (int): The index where CME features start in the data.
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

    # Plot the time-series data
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


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray or List[np.ndarray], y_test: np.ndarray) -> float:
    """
    Evaluates a given model using Mean Absolute Error (MAE) on the provided test data.

    Parameters:
    - model (tf.keras.Model): The trained model to evaluate.
    - X_test (np.ndarray): Test features.
    - y_test (np.ndarray): True target values for the test set.

    Returns:
    - float: The MAE loss of the model on the test data.
    """
    # Make predictions
    _, predictions = model.predict(X_test)

    # Calculate MAE
    mae_loss = mean_absolute_error(y_test, predictions)

    return mae_loss


def prepare_hybrid_inputs(
        data: np.ndarray,
        tsf_extractor_type: str = 'cnn',
        tsf_input_dims=None,
        with_slope: bool = False,
        mlp_input_dim: int = 23) -> Tuple[np.ndarray, ...]:
    """
    Splits the input data into parts for the TSF extractor branches (CNN or RNN) and a part for the MLP branch
    of the hybrid model.

    Parameters:
    - data (np.ndarray): The combined input data array.
    - tsf_extractor_type (str): The type of time series feature extractor ('cnn' or 'rnn').
    - tsf_input_dims (List[int]): The dimensions for each TSF extractor input.
    - with_slope (bool): Whether to add additional inputs for slopes.
    - mlp_input_dim (int): The number of features for the MLP branch.

    Returns:
    - Tuple: A tuple containing arrays for TSF extractor inputs and MLP input.
    """

    # Prepare inputs for the TSF extractor
    if tsf_extractor_type == 'cnn':
        tsf_inputs = prepare_cnn_inputs(data, tsf_input_dims, with_slope)
    elif tsf_extractor_type == 'rnn':
        tsf_inputs = prepare_rnn_inputs(data, tsf_input_dims, with_slope)
    else:
        raise ValueError("Invalid tsf_extractor_type. Must be 'cnn' or 'rnn'.")

    # Prepare inputs for the MLP branch
    # Assuming that the MLP input is located after the TSF input data
    start_index = sum(tsf_input_dims)
    mlp_input = data[:, start_index:start_index + mlp_input_dim]

    return *tsf_inputs, mlp_input


def prepare_cnn_inputs(data: np.ndarray, cnn_input_dims: List[int] = None, with_slope: bool = False) -> Tuple:
    """
    Splits the input data into parts for the CNN branches of the Y-shaped model,
    dynamically based on the cnn_input_dims list. If with_slope is True,
    the first half of cnn_input_dims are for regular inputs, and the second half are for slopes.

    Parameters:
    - data (np.ndarray): The combined input data array.
    - cnn_input_dims (List[int]): The dimensions for each CNN input.
    - with_slope (bool): Whether to add additional inputs for slopes.

    Returns:
    - Tuple: Tuple of arrays, each for CNN inputs.
    """
    if cnn_input_dims is None:
        cnn_input_dims = [25, 24]  # Default dimensions for regular and slope inputs

    # Initialize a list to store CNN inputs
    cnn_inputs = []

    # Split input dimensions into regular and slope inputs if with_slope is True
    if with_slope:
        half_len = len(cnn_input_dims) // 2
        regular_dims = cnn_input_dims[:half_len]
        slope_dims = cnn_input_dims[half_len:]
    else:
        regular_dims = cnn_input_dims
        slope_dims = []

    # Generate CNN inputs for regular data
    start_index = 0
    for dim in regular_dims:
        end_index = start_index + dim
        cnn_input = data[:, start_index:end_index].reshape((-1, dim, 1))
        cnn_inputs.append(cnn_input)
        start_index = end_index

    # Generate CNN inputs for slope data if with_slope is True
    for dim in slope_dims:
        end_index = start_index + dim
        slope_input = data[:, start_index:end_index].reshape((-1, dim, 1))
        cnn_inputs.append(slope_input)
        start_index = end_index

    return tuple(cnn_inputs)


# def prepare_rnn_inputs(data: np.ndarray, rnn_input_dims: List[int] = None, with_slope: bool = False) -> Tuple:
#     """
#     Splits the input data into parts for the RNN branches of the model,
#     dynamically based on the rnn_input_dims list. If with_slope is True,
#     the first half of rnn_input_dims are for regular inputs, and the second half are for slopes.
#
#     Parameters:
#     - data (np.ndarray): The combined input data array.
#     - rnn_input_dims (List[int]): The dimensions for each RNN input.
#     - with_slope (bool): Whether to add additional inputs for slopes.
#
#     Returns:
#     - Tuple: Tuple of arrays, each for RNN inputs.
#     """
#     if rnn_input_dims is None:
#         rnn_input_dims = [25, 24]  # Default dimensions for regular and slope inputs
#
#     # Initialize a list to store RNN inputs
#     rnn_inputs = []
#
#     # Split input dimensions into regular and slope inputs if with_slope is True
#     if with_slope:
#         half_len = len(rnn_input_dims) // 2
#         regular_dims = rnn_input_dims[:half_len]
#         slope_dims = rnn_input_dims[half_len:]
#     else:
#         regular_dims = rnn_input_dims
#         slope_dims = []
#
#     # Generate RNN inputs for regular data
#     start_index = 0
#     for dim in regular_dims:
#         end_index = start_index + dim
#         rnn_input = data[:, start_index:end_index, :]  # Assuming data is already 3D (samples, timesteps, features)
#         rnn_inputs.append(rnn_input)
#         start_index = end_index
#
#     # Generate RNN inputs for slope data if with_slope is True
#     for dim in slope_dims:
#         end_index = start_index + dim
#         slope_input = data[:, start_index:end_index, :]
#         rnn_inputs.append(slope_input)
#         start_index = end_index
#
#     return tuple(rnn_inputs)


def prepare_rnn_inputs(data: np.ndarray, rnn_input_dims: List[int] = None, with_slope: bool = False) -> Tuple:
    """
    Splits the input data into parts for the RNN branches of the model,
    dynamically based on the rnn_input_dims list. If with_slope is True,
    the first half of rnn_input_dims are for regular inputs, and the second half are for slopes.

    Parameters:
    - data (np.ndarray): The combined input data array.
    - rnn_input_dims (List[int]): The dimensions for each RNN input.
    - with_slope (bool): Whether to add additional inputs for slopes.

    Returns:
    - Tuple: Tuple of arrays, each for RNN inputs.
    """
    if rnn_input_dims is None:
        rnn_input_dims = [25, 24]  # Default dimensions for regular and slope inputs

    # Reshape data to 3D if it's 2D ([samples, features] -> [samples, timesteps=1, features])
    if data.ndim == 2:
        data = np.expand_dims(data, axis=1)

    # Initialize a list to store RNN inputs
    rnn_inputs = []

    # Split input dimensions into regular and slope inputs if with_slope is True
    if with_slope:
        half_len = len(rnn_input_dims) // 2
        regular_dims = rnn_input_dims[:half_len]
        slope_dims = rnn_input_dims[half_len:]
    else:
        regular_dims = rnn_input_dims
        slope_dims = []

    # Generate RNN inputs for regular data
    start_index = 0
    for dim in regular_dims:
        end_index = start_index + dim
        rnn_input = data[:, :, start_index:end_index]
        rnn_inputs.append(rnn_input)
        start_index = end_index

    # Generate RNN inputs for slope data if with_slope is True
    for dim in slope_dims:
        end_index = start_index + dim
        slope_input = data[:, :, start_index:end_index]
        rnn_inputs.append(slope_input)
        start_index = end_index

    return tuple(rnn_inputs)

