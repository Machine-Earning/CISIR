import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
# types for type hinting
from models import modeling
from sklearn.manifold import TSNE
import tensorflow as tf
import random
from datetime import datetime
from dataload import seploader as sepl

# SEEDING
SEED = 42  # seed number 

# Set NumPy seed
np.random.seed(SEED)

# Set TensorFlow seed
tf.random.set_seed(SEED)

# Set random seed
random.seed(SEED)


def count_above_threshold(y_values, threshold=np.log(10)):
    """
    Count the number of y values that are above a given threshold.

    Parameters:
    - y_values (array-like): The array of y-values to check.
    - threshold (float, optional): The threshold value to use for counting. Default is log(10).

    Returns:
    - int: The count of y-values that are above the threshold.
    """
    return np.sum(y_values > threshold)


def plot_tsne_and_save_with_timestamp(model, X, y, prefix, save_tag=None):
    """
    Applies t-SNE to the features extracted by the given model and saves the plot in 2D with a timestamp.
    The color of the points is determined by their label values.

    Parameters:
    - model: Trained feature extractor model
    - X: Input data (NumPy array or compatible)
    - y: Target labels (NumPy array or compatible)
    - prefix: Prefix for the file name

    Returns:
    - Saves a 2D t-SNE plot to a file with a timestamp
    """
    # saving the threshold
    threshold = np.log(9)
    # Extract features using the trained model
    features = model.predict(X)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=SEED)
    tsne_result = tsne.fit_transform(features)

    # Identify above and below threshold indices
    above_threshold_indices = np.where(y > threshold)[0]
    below_threshold_indices = np.where(y <= threshold)[0]

    # Create scatter plot for below-threshold points
    plt.figure(figsize=(12, 8))
    plt.scatter(tsne_result[below_threshold_indices, 0], tsne_result[below_threshold_indices, 1],
                c=y[below_threshold_indices], cmap='viridis', alpha=0.6, label='Below Threshold')

    # Overlay scatter plot for above-threshold points
    scatter = plt.scatter(tsne_result[above_threshold_indices, 0], tsne_result[above_threshold_indices, 1],
                          c=y[above_threshold_indices], cmap='viridis', alpha=1.0, edgecolors='r',
                          label='Above Threshold')

    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Label Value')

    # Add legend to differentiate above-threshold points
    plt.legend()

    plt.title('2D t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Save the plot
    file_path = f"{prefix}_tsne_plot_{str(save_tag)}.png"
    plt.savefig(file_path)
    plt.close()


def main():
    """
    Main function for testing the AI Panther
    :return: None
    """
    # check for gpus
    tf.config.list_physical_devices('GPU')
    # Read the CSV file
    loader = sepl.SEPLoader()
    shuffled_train_x, shuffled_train_y, shuffled_val_x, \
        shuffled_val_y, shuffled_test_x, shuffled_test_y = loader.load_from_dir('/home1/jmoukpe2016/keras-functional-api/cme_and_electron/data')

    train_count = count_above_threshold(shuffled_train_y)
    val_count = count_above_threshold(shuffled_val_y)
    test_count = count_above_threshold(shuffled_test_y)

    print(f'Training set: {train_count} above the threshold')
    print(f'Validation set: {val_count} above the threshold')
    print(f'Test set: {test_count} above the threshold')

    mb = modeling.ModelBuilder()

    # create my feature extractor
    feature_extractor = mb.create_model_feat(inputs=19, feat_dim=9, hiddens=[18])

    # load weights to continue training
    # feature_extractor.load_weights('model_weights_2023-09-28_18-25-47.h5')
    # print('weights loaded successfully!')

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # training
    Options = {
        'batch_size': 768,
        'epochs': 10000,
        'patience': 25,
        'learning_rate': 1e-3,
    }

    # print options used
    print(Options)
    mb.train_features(feature_extractor, shuffled_train_x, shuffled_train_y, shuffled_val_x, shuffled_val_y,
                      learning_rate=Options['learning_rate'],
                      epochs=Options['epochs'],
                      batch_size=Options['batch_size'],
                      patience=Options['patience'], save_tag=timestamp)

    # combine training and validation
    combined_train_x, combined_train_y = loader.combine(shuffled_train_x, shuffled_train_y, shuffled_val_x,
                                                        shuffled_val_y)

    plot_tsne_and_save_with_timestamp(feature_extractor, combined_train_x, combined_train_y, 'training',
                                      save_tag=timestamp)

    plot_tsne_and_save_with_timestamp(feature_extractor, shuffled_test_x, shuffled_test_y, 'testing',
                                      save_tag=timestamp)


if __name__ == '__main__':
    main()
