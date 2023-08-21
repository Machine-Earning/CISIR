# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
# for types hints
from typing import Tuple, Callable, List
from tensorflow import Tensor


def check_gpu():
    """
    Check for GPU availability
    :return: None
    """
    # check for gpus
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # list their names
    tf.config.experimental.list_physical_devices('GPU')


def load_and_preprocess_mnist() -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """
    Load the MNIST dataset, preprocess images, and perform one-hot encoding of labels.

    :return: Tuple of training data (x_train, y_train) and testing data (x_test, y_test).
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape and normalize images
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

    # One-hot encoding of labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def build_model() -> tf.keras.Model:
    """
    Build a simple MLP model for MNIST classification.
    :return: A tf.keras Model with inputs and outputs defined.
    """
    inputs = tf.keras.Input(shape=(784,), name='input')  # input layer
    x = tf.keras.layers.Dense(64, activation='relu', name='hidden1')(inputs)  # hidden layer
    x = tf.keras.layers.Dense(64, activation='relu', name='hidden2')(x)  # hidden layer
    outputs = tf.keras.layers.Dense(10, activation='softmax', name='output')(x)  # output layer

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def reciprocal_loss(y_true, y_pred, reduction=tf.keras.losses.Reduction.AUTO):
    """
    reciprocal loss for multi-class classification, tf.keras style.
    RL(p_t) = - 1/p_t * log(p_t), where p_t is the probability associated with the true class.

    :param y_true: Ground truth labels, shape of [batch_size, num_classes].
    :param y_pred: Predicted class probabilities, shape of [batch_size, num_classes].
    :param reduction: Reduction method to apply to the loss (default tf.keras.losses.Reduction.NONE).
    :return: A scalar representing the mean reciprocal loss over the batch.
    NOTE: written assuming GPU support to make use of fast Tensor operations.
    """
    # Create a Categorical Cross-Entropy loss instance
    cce = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE  # Keep unreduced loss tensor
    )
    # printing y_pred
    cross_entropy = cce(y_true, y_pred)  # batch_sizex1
    print(f'cross entropy after clipping: {cross_entropy}')
    _y_pred = y_pred
    # find the probability associated with the true class
    _y_true = K.argmax(y_true, axis=1)
    # get the predicted probability of the true class
    _y_pred = K.sum(_y_pred * y_true, axis=1)
    # reciprocal loss by dividing the cross entropy by the predicted probability of the true class
    _reciprocal_loss = cross_entropy / _y_pred  # reciprocal loss

    if reduction == tf.keras.losses.Reduction.AUTO:
        return K.mean(_reciprocal_loss)
    elif reduction == tf.keras.losses.Reduction.SUM:
        return K.sum(_reciprocal_loss)
    else:  # No reduction
        return _reciprocal_loss


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25, reduction=tf.keras.losses.Reduction.AUTO):
    """
        focal loss for multi-class classification, tf.keras style.
        RL(p_t) = - (1 - p_t) * log(p_t), where p_t is the probability associated with the true class.

        :param y_true: Ground truth labels, shape of [batch_size, num_classes].
        :param y_pred: Predicted class probabilities, shape of [batch_size, num_classes].
        :param reduction: Reduction method to apply to the loss (default tf.keras.losses.Reduction.NONE).
        :return: A scalar representing the mean focal loss over the batch.
        NOTE: written assuming GPU support to make use of fast Tensor operations.
        """
    # Create a Categorical Cross-Entropy loss instance
    cce = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE  # Keep unreduced loss tensor
    )
    # printing y_pred
    cross_entropy = cce(y_true, y_pred)  # batch_sizex1
    print(f'cross entropy after clipping: {cross_entropy}')
    _y_pred = y_pred
    # find the probability associated with the true class
    _y_true = K.argmax(y_true, axis=1)
    # get the predicted probability of the true class
    _y_pred = K.sum(_y_pred * y_true, axis=1)
    # focal loss by dividing the cross entropy by the predicted probability of the true class
    _focal_loss = alpha * K.pow(1 - _y_pred, gamma) * cross_entropy   # focal loss

    if reduction == tf.keras.losses.Reduction.AUTO:
        return K.mean(_focal_loss)
    elif reduction == tf.keras.losses.Reduction.SUM:
        return K.sum(_focal_loss)
    else:  # No reduction
        return _focal_loss


def create_imbalanced_data(x, y, imbalance_rate=0.5):
    """
    Create an imbalanced dataset based on a given probability distribution.
    The probability for class d is given by: P(d) = 0.5^d / 2*(1 - 0.5^10)

    :param x: Features, shape of [total_samples, feature_dim].
    :param y: One-hot encoded labels, shape of [total_samples, num_classes].
    :param imbalance_rate: Base rate for the exponential decay of class frequency (default 0.5).
    :return: Tuple of imbalanced features and labels, shapes of [selected_samples, feature_dim] and [selected_samples, num_classes].
    """
    total_samples = len(y)
    a = imbalance_rate
    normalization_factor = 2 * (1 - a ** 10)

    indices_by_class = [np.where(y[:, d] == 1)[0] for d in range(10)]
    selected_indices = []

    for d in range(10):
        probability_d = (a ** d) / normalization_factor
        frequency = int(total_samples * probability_d)
        np.random.shuffle(indices_by_class[d])  # Shuffle to ensure random selection
        selected_indices.extend(indices_by_class[d][:frequency])

    return x[selected_indices], y[selected_indices]


def accuracy_by_bins(model, x, y):
    """
    Calculate and print the accuracy of the given model for specific bins of classes.
    The bins are defined as: 0-1, 2-7, 8-9.

    :param model: Trained tf.keras model to evaluate.
    :param x: Input features, shape of [num_samples, feature_dim].
    :param y: One-hot encoded labels, shape of [num_samples, num_classes].
    """
    predictions = model.predict(x).argmax(axis=-1)
    true_labels = y.argmax(axis=-1)
    bins = [(0, 1), (2, 7), (8, 9)]
    for bin_start, bin_end in bins:
        mask = (true_labels >= bin_start) & (true_labels <= bin_end)
        bin_accuracy = np.mean(predictions[mask] == true_labels[mask])
        print(f"Accuracy for bin {bin_start}-{bin_end}: {bin_accuracy*100}")


def main():
    """
    Main function for testing the AI Panther
    :return: None
    """
    # check for gpus
    check_gpu()
    # load and preprocess mnist
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    # building our models for testing
    model_ce, model_fl, model_rl = build_model(), build_model(), build_model()
    # loading imbalance data
    x_train_imbalanced, y_train_imbalanced = create_imbalanced_data(x_train, y_train)
    # defining metrics
    metrics = [tf.keras.metrics.CategoricalAccuracy()]
    # compiling our models
    model_ce.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=metrics)
    model_fl.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=focal_loss,
        metrics=metrics)
    model_rl.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=3e-4),
        loss=reciprocal_loss,
        metrics=metrics)

    # training our models
    print("Training on imbalanced data:")
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    print("Cross Entropy:")
    model_ce.fit(
        x_train_imbalanced,
        y_train_imbalanced,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping_callback])

    print("Focal Loss:")
    model_fl.fit(
        x_train_imbalanced,
        y_train_imbalanced,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping_callback])

    print("reciprocal loss:")
    model_rl.fit(
        x_train_imbalanced,
        y_train_imbalanced,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping_callback])

    # Balanced data
    print("test on balanced set:")
    ce_res = model_ce.evaluate(x_test, y_test)
    fl_res = model_fl.evaluate(x_test, y_test)
    rl_res = model_rl.evaluate(x_test, y_test)
    print(f'ce: test loss {ce_res[0]}, test accuracy {ce_res[1] * 100}')
    print(f'fl: test loss {fl_res[0]}, test accuracy {fl_res[1] * 100}')
    print(f'rl: test loss {rl_res[0]}, test accuracy {rl_res[1] * 100}')
    # Imbalanced data

    print("test on imbalanced data:")
    x_test_imbalanced, y_test_imbalanced = create_imbalanced_data(x_test, y_test)
    ce_ires = model_ce.evaluate(x_test_imbalanced, y_test_imbalanced)
    fl_ires = model_fl.evaluate(x_test_imbalanced, y_test_imbalanced)
    rl_ires = model_rl.evaluate(x_test_imbalanced, y_test_imbalanced)
    print(f'ce: test loss {ce_ires[0]}, test accuracy {ce_ires[1] * 100}')
    print(f'fl: test loss {fl_ires[0]}, test accuracy {fl_ires[1] * 100}')
    print(f'rl: test loss {rl_ires[0]}, test accuracy {rl_ires[1] * 100}')

    print("Accuracy by bins for balanced data:")
    print("Cross Entropy:")
    accuracy_by_bins(model_ce, x_test, y_test)
    print("Focal Loss:")
    accuracy_by_bins(model_fl, x_test, y_test)
    print("reciprocal loss:")
    accuracy_by_bins(model_rl, x_test, y_test)

    print("Accuracy by bins for imbalanced data:")
    print("Cross Entropy:")
    accuracy_by_bins(model_ce, x_test_imbalanced, y_test_imbalanced)
    print("Focal Loss:")
    accuracy_by_bins(model_fl, x_test_imbalanced, y_test_imbalanced)
    print("reciprocal loss:")
    accuracy_by_bins(model_rl, x_test_imbalanced, y_test_imbalanced)


if __name__ == '__main__':
    main()
