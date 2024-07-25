import os
from datetime import datetime
from typing import Tuple

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd
import wandb
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from wandb.keras import WandbCallback
import random

# Importing the Blocks
from sources.transformer.modules import *


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


def generate_unique_dataset(n_points: int, exclude_set: set) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset ensuring no overlap with the exclude set.

    Args:
        n_points (int): Number of data points to generate.
        exclude_set (set): Set of tuples representing points to exclude.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Generated features and labels.
    """
    np.random.seed(0)
    unique_points = set()
    x1_list, x2_list, y_list = [], [], []

    while len(unique_points) < n_points:
        # Generate x1 and x2
        x1_int = np.random.randint(-5, 6, size=n_points // 2)
        x1_float = np.random.uniform(-5, 5, size=n_points - n_points // 2)
        x1 = np.concatenate([x1_int, x1_float])

        x2_int = np.random.randint(-5, 6, size=n_points // 2)
        x2_float = np.random.uniform(-5, 5, size=n_points - n_points // 2)
        x2 = np.concatenate([x2_int, x2_float])

        # Shuffle the arrays to mix integers and floats
        np.random.shuffle(x1)
        np.random.shuffle(x2)

        for xi, xj in zip(x1, x2):
            if len(unique_points) >= n_points:
                break
            point = (round(xi, 2), round(xj, 2))
            if point not in exclude_set:
                unique_points.add(point)
                x1_list.append(xi)
                x2_list.append(xj)
                y = xi if xj < 0 else xi + xj
                y_list.append(y)

    return np.stack((x1_list, x2_list), axis=1), np.array(y_list)


# Initial test set
initial_x = np.array([
    [1, -1], [2, 1], [3, -3],
    [4, 5], [-1, -1], [-3, 2],
    [-5, 5], [-4, -5], [0, 0],
    [0, 4]
])
initial_y = np.array([
    1, 3, 3,
    9, -1, -1,
    0, -4, 0,
    4
])

# Convert initial_x to a set of tuples for exclusion
initial_set = set((round(xi[0], 2), round(xi[1], 2)) for xi in initial_x)

# Generate training set ensuring no overlap with initial set
x_train, y_train = generate_unique_dataset(5000, initial_set)

# Combine initial set into the exclusion set for generating the final test set
train_set = set((round(xi[0], 2), round(xi[1], 2)) for xi in x_train)
total_exclude_set = initial_set.union(train_set)

# Generate test set ensuring no overlap with training set and initial set
x_test, y_test = generate_unique_dataset(1000, total_exclude_set)

# Verify no overlap
assert not set(map(tuple, x_test)).intersection(set(map(tuple, x_train)))
assert not set(map(tuple, x_test)).intersection(initial_set)

print("Training and test sets generated without overlap.")


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
        verbose=0,
        validation_data=(x_test, {'output': y_test}),
        callbacks=[early_stopping, WandbCallback(save_model=False)]
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
                [inp[0], inp[1], true, pred[0]]
                + attn.tolist()
                + attention_weighted_values
                + [dense_layer_bias[0]]
                + dense_layer_weights[:, 0].tolist()
            )

        # Print results in a table
        headers = (
                ['x1', 'x2', 'True y', 'Predicted y']
                + [f'Attention_{i + 1}' for i in range(attention_scores.shape[1])]
                + [f'Attention_Weight_{i + 1}' for i in range(attention_scores.shape[1])]
                + ['Bias'] + [f'Weight_{i + 1}' for i in range(dense_layer_weights.shape[0])]
        )
        df_results = pd.DataFrame(results, columns=headers)
        print(df_results)
        wandb.log({f"results_{block_name}": df_results})  # so cool you can log dataframes


# Training and printing results for each attention type
input_shape = (2,)
block_classes = [BlockT0, BlockT1, BlockT2, BlockT3, BlockT4, BlockT5, BlockT6]

for i, block_class in enumerate(block_classes):
    if i not in [3]:
        continue  # skip the first 4
    # Create a unique experiment name with a timestamp
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f'Attention_Type{i}_{current_time}'
    # Initialize wandb
    LR = 1e-3
    EPOCHS = int(5e3)
    BS = 256
    PATIENCE = 250

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
        x_debug=initial_x, y_debug=initial_y,
        learning_rate=LR, epochs=EPOCHS,
        batch_size=BS, patience=PATIENCE
    )

    # Finish the wandb run
    wandb.finish()
