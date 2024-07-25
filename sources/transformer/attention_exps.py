import os
from typing import Tuple
from datetime import datetime

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from wandb.keras import WandbCallback

# Importing the Blocks
from sources.transformer.modules import BlockT1, BlockT2, BlockT3, BlockT4, BlockT5, BlockT6


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
    np.random.seed(42)

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
        initial_x: np.ndarray,
        initial_y: np.ndarray,
        learning_rate: float = 0.003,
        epochs: int = 500,
        batch_size: int = 32) -> None:
    """
    Train the model and print the results, including learned weights and attention scores.

    Args:
        block_name (str): Name of the attention block.
        model (Model): The Keras model to train.
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        x_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        initial_x (np.ndarray): Initial test features for detailed output.
        initial_y (np.ndarray): Initial test labels for detailed output.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), 
        loss={'output': 'mse'}, 
        metrics={'output': 'mae'}
    )
                  
    
    model.fit(
        x_train,
        {'output': y_train},
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_data=(x_test,{'output': y_test}),
        callbacks=[WandbCallback(save_model=False)]
    )

     # Evaluate the model - focus on the 'output' key
    loss, mae = model.evaluate(x_test, {'output': y_test}, verbose=0)
    print(f"Test loss: {loss}")
    print(f"MAE loss: {mae}")

    # Predict on initial data
    predictions = model.predict(initial_x)
    output_predictions = predictions['output']
    attention_scores = predictions['attention_scores']
    print("Predictions on initial data:")

    results = []
    for pred, true, inp, attn in zip(output_predictions, initial_y, initial_x, attention_scores):
        results.append([inp[0], inp[1], true, pred[0]] + attn.tolist())

    # Print results in a table
    headers = (['x1', 'x2', 'True y', 'Predicted y'] +
               [f'Attention_{i + 1}' for i in range(attention_scores.shape[1])])
    df_results = pd.DataFrame(results, columns=headers)
    print(df_results)
    wandb.log({"results": df_results})


# Training and printing results for each attention type
input_shape = (2,)
block_classes = [BlockT1, BlockT2, BlockT3, BlockT4, BlockT5, BlockT6]

for i, block_class in enumerate(block_classes, start=1):
    # Create a unique experiment name with a timestamp
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f'Attention_Type{i}_{current_time}'
    # Initialize wandb
    wandb.init(project="attention-exps", name=experiment_name)
    print(f"\nAttention Type {i}")
    model = create_model(block_class, input_shape)
    model.summary()
    # tf.keras.utils.plot_model(model, to_file=f'./model_{i}.png', show_shapes=True)
    train_and_print_results(
        f'Attention Typ {i}',
        model, x_train, y_train, x_test, y_test,
        initial_x, initial_y,
        learning_rate=0.001,
        epochs=2000, batch_size=32)
