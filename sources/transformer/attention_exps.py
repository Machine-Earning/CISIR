import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
import os

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def generate_dataset(n_points: int) -> (np.ndarray, np.ndarray):
    """
    Generate a synthetic dataset based on the given rules.
    If x2 is negative, y = 2 * x1.
    If x2 is positive, y = x1 + x2.

    Args:
        n_points (int): Number of data points to generate.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Generated features and labels.
    """
    np.random.seed(42)
    x1 = np.random.randint(1, 10, size=n_points)
    x2 = np.random.randint(-5, 6, size=n_points)
    y = np.array([2 * x1[i] if x2[i] < 0 else x1[i] + x2[i] for i in range(n_points)])
    return np.stack((x1, x2), axis=1), y

# Initial given data
initial_x = np.array([[1, -1], [2, 1], [3, -3], [4, 5]])
initial_y = np.array([2, 3, 6, 9])

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

def train_and_print_results(model: Model, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, epochs: int = 500) -> None:
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
    get_attention_model = Model(inputs=model.input, outputs=[layer.output for layer in model.layers if 'dense' in layer.name])
    attention_outputs = get_attention_model.predict(initial_x)
    
    print("Attention outputs on initial data:")
    for i, output in enumerate(attention_outputs):
        print(f"Layer {i+1} output:\n{output}")

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
