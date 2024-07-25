import os
from datetime import datetime
from typing import Tuple

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from wandb.keras import WandbCallback
import random

# Importing the Blocks
from sources.transformer.modules import BlockT1, BlockT2, BlockT3, BlockT4, BlockT5, BlockT6


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


def generate_dataset(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset based on the given rules.
    If x2 is negative, y = 2 * x1.
    If x2 is positive or zero, y = x1 + x2.

    Args:
        n_points (int): Number of data points to generate.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Generated features and labels.
    """
    np.random.seed(0)

    # Generate x1: mix of integers and floats, positive and negative
    x1_int = np.random.randint(-5, 6, size=n_points // 2)
    x1_float = np.random.uniform(-5, 5, size=n_points - n_points // 2)
    x1 = np.concatenate([x1_int, x1_float])

    # Generate x2: mix of integers and floats, positive and negative
    x2_int = np.random.randint(-5, 6, size=n_points // 2)
    x2_float = np.random.uniform(-5, 5, size=n_points - n_points // 2)
    x2 = np.concatenate([x2_int, x2_float])

    # Shuffle the arrays to mix integers and floats
    np.random.shuffle(x1)
    np.random.shuffle(x2)

    # Generate y based on the rule
    y = np.where(x2 < 0, 2 * x1, x1 + x2)

    return np.stack((x1, x2), axis=1), y


# Test set
initial_x = np.array([
    [1, -1], [2, 1], [3, -3],
    [4, 5], [-1, -1], [-3, 2],
    [-5, 5], [-4, -5], [0, 0],
    [0, 4]
])
initial_y = np.array([
    2, 3, 6,
    9, -2, -1,
    0, -9, 0,
    4
])

# Generate additional data points for training
x_train, y_train = generate_dataset(1000)
x_test, y_test = initial_x, initial_y


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
        learning_rate: float = 0.003,
        epochs: int = 500,
        batch_size: int = 32,
        patience: int = 100) -> None:
    """
    Train the model and print the results, including learned weights and attention scores.

    Args:
        block_name (str): Name of the attention block.
        model (Model): The Keras model to train.
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        x_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
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

    # Predict on initial data
    predictions = model.predict(x_test)
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
    # results = []
    # for pred, true, inp, attn in zip(output_predictions, y_test, x_test, attention_scores):
    #     results.append([inp[0], inp[1], true, pred[0]] + attn.tolist())
    #
    # # Print results in a table
    # headers = (['x1', 'x2', 'True y', 'Predicted y'] +
    #            [f'Attention_{i + 1}' for i in range(attention_scores.shape[1])])
    # df_results = pd.DataFrame(results, columns=headers)
    # print(df_results)
    # wandb.log({"results": df_results})  # so coool you can log dataframes

    results = []
    for pred, true, inp, attn in zip(output_predictions, initial_y, initial_x, attention_scores):
        results.append([
            inp[0], inp[1], true, pred[0]
            ] + attn.tolist() + [
                dense_layer_bias[0]
            ] + dense_layer_weights[:,0].tolist()
        )

    # Print results in a table
    headers = (['x1', 'x2', 'True y', 'Predicted y'] +
               [f'Attention_{i + 1}' for i in range(attention_scores.shape[1])] +
               ['Bias'] + [f'Weight_{i + 1}' for i in range(dense_layer_weights.shape[0])])
    df_results = pd.DataFrame(results, columns=headers)
    print(df_results)
    wandb.log({"results": df_results})  # so cool you can log dataframes


# Training and printing results for each attention type
input_shape = (2,)
block_classes = [BlockT1, BlockT2, BlockT3, BlockT4, BlockT5, BlockT6]

for i, block_class in enumerate(block_classes, start=1):
    # Create a unique experiment name with a timestamp
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f'Attention_Type{i}_{current_time}'
    # Initialize wandb
    LR = 1e-3
    EPOCHS = 2000
    BS = 32
    PATIENCE = 200

    wandb.init(project="attention-exps", name=experiment_name, config={
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
        str(i), model, x_train, y_train, x_test, y_test,
        learning_rate=LR, epochs=EPOCHS, batch_size=BS, patience=PATIENCE
    )

    # Finish the wandb run
    wandb.finish()
