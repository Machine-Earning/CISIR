import os
from typing import List, Optional

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Multiply
from tensorflow.keras.models import Model

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def generate_dataset(n_points: int) -> (np.ndarray, np.ndarray):
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


# Initial given data
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

# Generate additional data points
x_additional, y_additional = generate_dataset(10)

# Combine initial and additional data
x = np.vstack((initial_x, x_additional))
y = np.concatenate((initial_y, y_additional))

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Training Dataset X:\n", x_train)
print("Training Dataset Y:\n", y_train)
print("Test Dataset X:\n", x_test)
print("Test Dataset Y:\n", y_test)


class AttentionLayer(Layer):
    """
    A custom attention layer that applies a series of dense layers followed by an output layer.

    This layer can be used to compute attention scores or feature transformations
    in attention-based neural network architectures.

    Attributes:
        input_dim (int): The dimensionality of the input.
        hidden_units (List[int]): A list of integers, where each integer is the number of units
                                  in the corresponding dense layer.
        output_dim (int): The dimensionality of the output.
        activation (str): The activation function to use in the hidden layers.
        dense_layers (List[Dense]): A list of Dense layers used in the attention mechanism.
        output_layer (Dense): The final Dense layer that produces the output.
    """

    def __init__(self, input_dim: int, hidden_units: List[int], output_dim: int, activation: str = 'tanh'):
        """
        Initialize the AttentionLayer.

        Args:
            input_dim (int): The dimensionality of the input.
            hidden_units (List[int]): A list of integers, where each integer is the number of units
                                      in the corresponding dense layer.
            output_dim (int): The dimensionality of the output.
            activation (str, optional): The activation function to use in the hidden layers.
                                        Defaults to 'tanh'.
        """
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.activation = activation
        self.dense_layers: List[Dense] = []

        # Create dense layers based on hidden_units list
        for units in hidden_units:
            self.dense_layers.append(Dense(units, activation=self.activation))
        self.output_layer = Dense(output_dim)

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer. This method is called automatically by Keras when the layer is first used.

        In this implementation, the layers are created in __init__, so this method is empty.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        """
        # The layers are created in __init__, so no need to create them here.
        pass

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Perform the forward pass of the attention layer.

        This method applies a series of dense layers to the input, followed by a final output layer.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor after applying the attention mechanism.
        """
        x = inputs
        # Apply each dense layer in sequence
        for dense in self.dense_layers:
            x = dense(x)
        # Apply the final output layer
        return self.output_layer(x)


class BlockT1(Layer):
    """
    A custom layer that applies attention mechanism followed by a dense layer.

    This block consists of an attention layer that computes attention scores,
    applies these scores to the input via element-wise multiplication,
    and then passes the result through a dense layer.

    Attributes:
        attn_hidden_units (List[int]): A list of integers representing the number of units
                                       in each hidden layer of the attention mechanism.
        activation (str): The activation function to use in the attention layer.
        output_activation (Optional[str]): The activation function to use in the final dense layer.
        attention_layer (AttentionLayer): The layer used to compute attention scores.
        dense_layer (Dense): The final dense layer that produces the output.
    """

    def __init__(self,
                 attn_hidden_units: Optional[List[int]] = None,
                 activation: str = 'tanh',
                 output_activation: Optional[str] = None):
        """
        Initialize the BlockT1 layer.

        Args:
            attn_hidden_units (Optional[List[int]]): A list of integers representing the number
                                                     of units in each hidden layer of the
                                                     attention mechanism. Defaults to [3].
            activation (str): The activation function to use in the attention layer. 
                              Defaults to 'tanh'.
            output_activation (Optional[str]): The activation function to use in the final 
                                               dense layer. Defaults to None.
        """
        super(BlockT1, self).__init__()
        self.dense_layer = None
        self.attention_layer = None
        self.attn_hidden_units = attn_hidden_units or [3]
        self.activation = activation
        self.output_activation = output_activation

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer. This method is called automatically by Keras when the layer is first used.

        Args:
            input_shape (tf.TensorShape): The shape of the input tensor.
        """
        # Create the attention layer
        self.attention_layer = AttentionLayer(
            input_dim=input_shape[-1],
            hidden_units=self.attn_hidden_units,
            output_dim=input_shape[-1],
            activation=self.activation
        )
        # Create the final dense layer
        self.dense_layer = Dense(1, activation=self.output_activation)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Perform the forward pass of the BlockT1 layer.

        This method applies the attention mechanism to the inputs, 
        then passes the weighted inputs through a dense layer.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor after applying attention and the dense layer.
        """
        # Compute attention scores
        attention_scores = self.attention_layer(inputs)

        # Apply attention scores to inputs via element-wise multiplication
        weighted_inputs = Multiply()([inputs, attention_scores])

        # Pass the weighted inputs through the final dense layer
        output = self.dense_layer(weighted_inputs)

        return output


# Example usage:
# block_t1 = BlockT1(attn_hidden_units=[3, 3], activation='tanh', output_activation=None)
# output = block_t1(inputs)


def create_attention_type1(input_shape: tuple) -> Model:
    """
    Create a model where the output is a linear function with learnable weights and
    attention scores calculated by small MLPs.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        Model: A Keras Model for the specified attention mechanism.
    """
    inputs = Input(shape=input_shape)
    x1, x2 = inputs[:, 0], inputs[:, 1]

    # Define the MLPs for attention with one hidden layer of 3 neurons
    a1_hidden = Dense(3, activation='relu')(inputs)
    a1 = Dense(1, activation='relu')(a1_hidden)
    a2_hidden = Dense(3, activation='relu')(inputs)
    a2 = Dense(1, activation='relu')(a2_hidden)

    # Calculate y
    w0 = tf.Variable(0.0)
    w1 = tf.Variable(0.05)
    w2 = tf.Variable(0.05)

    # 
    y = Dense(1)()

    y = w0 + w1 * a1 * x1 + w2 * a2 * x2
    model = Model(inputs, y)

    return model


def create_attention_type2(input_shape: tuple) -> Model:
    """
    Create a model where the output is a weighted sum of inputs,
    with attention scores calculated by small MLPs.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        Model: A Keras Model for the specified attention mechanism.
    """
    inputs = Input(shape=input_shape)
    x1, x2 = inputs[:, 0], inputs[:, 1]

    # Define the MLPs for attention with one hidden layer of 3 neurons
    a1_hidden = Dense(3, activation='relu')(inputs)
    a1 = Dense(1, activation='relu')(a1_hidden)
    a2_hidden = Dense(3, activation='relu')(inputs)
    a2 = Dense(1, activation='relu')(a2_hidden)

    # Calculate y
    y = a1 * x1 + a2 * x2
    model = Model(inputs, y)

    return model


def create_attention_type3(input_shape: tuple) -> Model:
    """
    Create a model where the output is a linear function with learnable weights plus
    attention scores calculated by small MLPs.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        Model: A Keras Model for the specified attention mechanism.
    """
    inputs = Input(shape=input_shape)
    x1, x2 = inputs[:, 0], inputs[:, 1]

    # Define the MLPs for attention with one hidden layer of 3 neurons
    a1_hidden = Dense(3, activation='relu')(inputs)
    a1 = Dense(1, activation='relu')(a1_hidden)
    a2_hidden = Dense(3, activation='relu')(inputs)
    a2 = Dense(1, activation='relu')(a2_hidden)

    # Calculate y
    w0 = tf.Variable(0.0)
    w1 = tf.Variable(0.05)
    w2 = tf.Variable(0.05)

    y = w0 + (w1 + a1) * x1 + (w2 + a2) * x2
    model = Model(inputs, y)

    return model


def create_attention_type4(input_shape: tuple) -> Model:
    """
    Create a model where the output is a linear function with learnable weights and
    attention scores calculated by small MLPs with softmax activation.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        Model: A Keras Model for the specified attention mechanism.
    """
    inputs = Input(shape=input_shape)
    x1, x2 = inputs[:, 0], inputs[:, 1]

    # Define the MLPs for attention with one hidden layer of 3 neurons
    a1_hidden = Dense(3, activation='relu')(inputs)
    a1 = Dense(1, activation='softmax')(a1_hidden)
    a2_hidden = Dense(3, activation='relu')(inputs)
    a2 = Dense(1, activation='softmax')(a2_hidden)

    # Calculate y
    w0 = tf.Variable(0.0)
    w1 = tf.Variable(0.05)
    w2 = tf.Variable(0.05)

    y = w0 + w1 * a1 * x1 + w2 * a2 * x2
    model = Model(inputs, y)

    return model


def create_attention_type5(input_shape: tuple) -> Model:
    """
    Create a model where the output is a linear function with learnable weights and
    attention scores calculated by small MLPs with sigmoid activation.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        Model: A Keras Model for the specified attention mechanism.
    """
    inputs = Input(shape=input_shape)
    x1, x2 = inputs[:, 0], inputs[:, 1]

    # Define the MLPs for attention with one hidden layer of 3 neurons
    a1_hidden = Dense(3, activation='relu')(inputs)
    a1 = Dense(1, activation='sigmoid')(a1_hidden)
    a2_hidden = Dense(3, activation='relu')(inputs)
    a2 = Dense(1, activation='sigmoid')(a2_hidden)

    # Calculate y
    w0 = tf.Variable(0.0)
    w1 = tf.Variable(0.05)
    w2 = tf.Variable(0.05)

    y = w0 + w1 * a1 * x1 + w2 * a2 * x2
    model = Model(inputs, y)

    return model


def create_attention_type6(input_shape: tuple) -> Model:
    """
    Create a model where the output is a linear function with learnable weights and
    attention scores calculated by small MLPs with sigmoid activation,
    normalized to sum to 1.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        Model: A Keras Model for the specified attention mechanism.
    """
    inputs = Input(shape=input_shape)
    x1, x2 = inputs[:, 0], inputs[:, 1]

    # Define the MLPs for attention with one hidden layer of 3 neurons
    a1_hidden = Dense(3, activation='relu')(inputs)
    a1 = Dense(1, activation='sigmoid')(a1_hidden)
    a2_hidden = Dense(3, activation='relu')(inputs)
    a2 = Dense(1, activation='sigmoid')(a2_hidden)

    # Normalize attention scores
    def normalize_attention_scores(inputs):
        a1, a2 = inputs
        sum_attentions = a1 + a2
        return a1 / sum_attentions, a2 / sum_attentions

    a1, a2 = Lambda(normalize_attention_scores)([a1, a2])

    # Calculate y
    w0 = tf.Variable(0.0)
    w1 = tf.Variable(0.05)
    w2 = tf.Variable(0.05)

    y = w0 + w1 * a1 * x1 + w2 * a2 * x2
    model = Model(inputs, y)

    return model


def train_and_print_results(
        model: Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 500) -> None:
    """
    Train the model and print the results, including learned weights and attention scores.

    Args:
        model (Model): The Keras model to train.
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        x_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        epochs (int): Number of epochs to train the model.
    """
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train, epochs=epochs, verbose=0, validation_data=(x_test, y_test))

    # Print learned weights and attention scores
    for layer in model.layers:
        if 'dense' in layer.name:
            print(f"Layer {layer.name} weights: {layer.get_weights()}")
        elif 'variable' in layer.name:
            print(f"Layer {layer.name} variable: {layer.numpy()}")

    # Evaluate the model
    loss = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss}")

    # Predict on initial data
    predictions = model.predict(initial_x)
    print("Predictions on initial data:")
    for pred, true in zip(predictions, initial_y):
        print(f"Prediction: {pred[0]:.4f}, True label: {true}")

    # Get attention scores
    get_attention_model = Model(inputs=model.input,
                                outputs=[layer.output for layer in model.layers if 'dense' in layer.name])
    attention_outputs = get_attention_model.predict(initial_x)

    print("Attention outputs on initial data:")
    for i, output in enumerate(attention_outputs):
        print(f"Layer {i + 1} output:\n{output}")


# Training and printing results for each attention type
input_shape = (2,)
models = [
    create_attention_type1(input_shape),
    create_attention_type2(input_shape),
    create_attention_type3(input_shape),
    create_attention_type4(input_shape),
    create_attention_type5(input_shape),
    create_attention_type6(input_shape),
]

for i, model in enumerate(models, start=1):
    print(f"\nAttention Type {i}")
    model.summary()
    tf.keras.utils.plot_model(model, to_file=f'./model_{i}.png', show_shapes=True)
    train_and_print_results(model, x_train, y_train, x_test, y_test)
