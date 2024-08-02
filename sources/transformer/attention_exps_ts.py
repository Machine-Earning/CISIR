import os
import random
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import wandb
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from wandb.integration.keras import WandbCallback

# Importing the Blocks
from sources.transformer.modules import *

# Set the environment variable for CUDA (in case it is necessary)
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# TODO: double check and update like he asked so everything looks good

devices = tf.config.list_physical_devices('GPU')
print(f'devices: {devices}')


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


set_seed(0)  # Set seed for reproducibility


def create_event_series(event_type: str) -> np.ndarray:
    """
    Create a time series for a specific event type.

    Args:
        event_type (str): Type of the event ('fast' or 'slow').

    Returns:
        np.ndarray: Time series data.
    """
    series = np.ones(30)
    if event_type == 'fast':
        series[10:20] = np.linspace(1, 10, 10)  # Increase by 1 for 10 timestamps
    elif event_type == 'slow':
        series[10:20] = np.linspace(1, 3, 10)  # Increase by 0.2 for 10 timestamps
    series[20:] = series[19]  # Plateau after rise
    return series


def generate_time_series_data(n_samples: int, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic time series data for fast rising and slow rising events.

    Args:
        n_samples (int): Number of samples to generate for each type.
        shuffle (bool): Whether to shuffle the generated data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Generated features and labels.
    """
    x_list, y_list = [], []

    for _ in range(n_samples):
        for event_type in ['fast', 'slow']:
            series = create_event_series(event_type)
            for t in range(4, 30):
                x_list.append(series[t - 4:t + 1])
                y_list.append(series[min(t + 3, 29)])  # Ensure we don't go out of bounds

    x_array = np.array(x_list)
    y_array = np.array(y_list)

    if shuffle:
        indices = np.arange(x_array.shape[0])
        np.random.shuffle(indices)
        x_array = x_array[indices]
        y_array = y_array[indices]

    return x_array, y_array


def select_debug_samples(x_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select debug samples from background, rising edge, and plateau.

    Args:
        x_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Selected debug features and labels.
    """
    background_samples = []
    rising_edge_samples = []
    plateau_samples = []

    for x, y in zip(x_test, y_test):
        if x[-1] == 1:
            background_samples.append((x, y))
        elif 1 < x[-1] < 10:
            rising_edge_samples.append((x, y))
        elif x[-1] == 10 or x[-1] == 3:
            plateau_samples.append((x, y))

    debug_samples = (
            background_samples[:2] +
            rising_edge_samples[:2] +
            plateau_samples[:2]
    )

    x_debug = np.array([x for x, y in debug_samples])
    y_debug = np.array([y for x, y in debug_samples])

    return x_debug, y_debug


# Generate training and test data
n_samples_train = 500
n_samples_test = 100

x_train, y_train = generate_time_series_data(n_samples_train)
x_test, y_test = generate_time_series_data(n_samples_test)

# Select 6 instances for debugging
x_debug, y_debug = generate_time_series_data(1, shuffle=False) 
#select_debug_samples(x_test, y_test)

# Verify data shapes
print(f"Shape of x_train: {x_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of x_test: {x_test.shape}")
print(f"Shape of y_test: {y_test.shape}")
print(f"Shape of x_debug: {x_debug.shape}")
print(f"Shape of y_debug: {y_debug.shape}")


def create_model(block_class, input_shape: Tuple[int]) -> Model:
    """
    Create a model using the specified block class.

    Args:
        block_class: The block class to use.
        input_shape (Tuple[int]): The shape of the input tensor.

    Returns:
        Model: The Keras model.
    """
    inputs = Input(shape=input_shape)
    block = block_class(attn_hidden_units=[4, 4], activation='tanh', output_activation='linear')
    outputs = block(inputs)  # dict of outputs and attention scores
    model = Model(inputs, outputs=outputs)
    return model


def train_and_print_results(
        block_name: str,
        model: Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        x_debug: np.ndarray = None,
        y_debug: np.ndarray = None,
        learning_rate: float = 0.003,
        epochs: int = 500,
        batch_size: int = 32,
        patience: int = 100,
        debug: bool = True) -> None:
    """
    Train the model and print the results, including learned weights and attention scores.

    Args:
        block_name (str): Name of the attention block.
        model (Model): The Keras model to train.
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        x_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        x_debug (np.ndarray): Debug features.
        y_debug (np.ndarray): Debug labels.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training.
        patience (int): Patience for early stopping.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={'output': 'mse'},
        metrics={'output': 'mae'}
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    model.fit(
        x_train,
        {'output': y_train},
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(x_test, {'output': y_test}),
        callbacks=[early_stopping, WandbCallback(save_model=False)],
    )

    # Evaluate the model - focus on the 'output' key
    results = model.evaluate(x_test, {'output': y_test}, verbose=0, return_dict=True)
    print('results')
    print(results)
    loss = results['loss']
    mae = results[f'block_t{block_name}_1_mae']
    print(f"Test loss: {loss}")
    print(f"MAE loss: {mae}")
    # log the results
    wandb.log({"loss": loss, "mae": mae})

    # Evaluate the model on the debug set
    results = model.evaluate(x_debug, {'output': y_debug}, verbose=0, return_dict=True)
    print('results')
    print(results)
    loss = results['loss']
    mae = results[f'block_t{block_name}_1_mae']
    print(f"Debug loss: {loss}")
    print(f"Debug MAE loss: {mae}")
    # log the results
    wandb.log({"debug_loss": loss, "debug_mae": mae})

    if debug:
        # Predict on initial data
        predictions = model.predict(x_debug)
        output_predictions = predictions['output']
        attention_scores = predictions['attention_scores']
        print("Predictions on initial data:")

        if block_name != '2':
            # Retrieve the weights and bias of the last dense layer
            dense_layer_weights, dense_layer_bias = model.layers[-1].get_dense_weights()
            print('weights')
            print(dense_layer_weights)
            print(dense_layer_bias)
        else:
            dense_layer_weights, dense_layer_bias = np.array([[1], [1]]), np.array([0])

        results = []
        for pred, true, inp, attn in zip(output_predictions, y_debug, x_debug, attention_scores):
            attention_weighted_values = [a * w for a, w in zip(attn, dense_layer_weights[:, 0])]
            results.append(
                list(inp) + [true, pred[0]]
                + attn.tolist()
                + attention_weighted_values
                + [dense_layer_bias[0]]
                + dense_layer_weights[:, 0].tolist()
            )

        # Print results in a table
        headers = (
                [f'x{i + 1}' for i in range(len(inp))]
                + ['True y', 'Predicted y']
                + [f'Attention_{i + 1}' for i in range(attention_scores.shape[1])]
                + [f'Attention_Weight_{i + 1}' for i in range(attention_scores.shape[1])]
                + ['Bias'] + [f'Weight_{i + 1}' for i in range(dense_layer_weights.shape[0])]
        )

        df_results = pd.DataFrame(results, columns=headers)
        print(df_results)
        wandb.log({f"results_{block_name}": df_results})  # so cool you can log dataframes


# Training and printing results for each attention type
input_shape = (5,)
block_classes = [BlockT0, BlockT1, BlockT2, BlockT3, BlockT4, BlockT5, BlockT6, BlockT7]

for i, block_class in enumerate(block_classes):
    if i not in [0, 7]:
        continue  # skip the first 4
    # Create a unique experiment name with a timestamp
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f'Attention_Type{i}_{current_time}'
    # Initialize wandb
    LR = 3e-3
    EPOCHS = int(5e3)
    BS = 256
    PATIENCE = 500

    wandb.init(project="attention-exps-2", name=experiment_name, config={
        "learning_rate": LR,
        "epochs": EPOCHS,
        "batch_size": BS,
        "attention_type": i,
        "patience": PATIENCE
    })

    print(f"\nAttention Type {i}")
    model = create_model(block_class, input_shape)
    model.summary()
    # tf.keras.utils.plot_model(model, to_file=f'./model_{i}.png', show_shapes=True)
    train_and_print_results(
        str(i), model,
        x_train, y_train,
        x_test, y_test,
        x_debug=x_debug, y_debug=y_debug,
        learning_rate=LR, epochs=EPOCHS,
        batch_size=BS, patience=PATIENCE
    )

    # Finish the wandb run
    wandb.finish()
