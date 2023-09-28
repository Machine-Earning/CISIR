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


def split_data(df):
    """
    Splits the data into training, validation, and test sets according to the specified rules.

    :param:
    - df: DataFrame containing the data. Assumes 'log_peak_intensity' is the target column.

    :return:
    - train_x, train_y, val_x, val_y, test_x, test_y: Numpy arrays containing the split data
    """
    # Sort the DataFrame by 'log_peak_intensity' in descending order
    df_sorted = df.sort_values(by='log_peak_intensity', ascending=False).reset_index(drop=True)

    # Initialize empty lists to store indices for training, validation, and test sets
    train_indices = []
    val_indices = []
    test_indices = []

    # Group every 3 rows for test set selection
    for i in range(0, len(df_sorted), 3):
        group = list(range(i, min(i + 3, len(df_sorted))))
        test_idx = np.random.choice(group, 1)[0]
        train_indices.extend([idx for idx in group if idx != test_idx])
        test_indices.append(test_idx)

    # Group every 4 rows for validation set selection from the training set
    for i in range(0, len(train_indices), 4):
        group = train_indices[i: i + 4]
        if len(group) == 0:
            continue
        val_idx = np.random.choice(group, 1)[0]
        val_indices.append(val_idx)
        train_indices = [idx for idx in train_indices if idx != val_idx]

    # Extract the feature and target sets based on selected indices
    features = df_sorted.drop(columns=['log_peak_intensity']).to_numpy()
    target = df_sorted['log_peak_intensity'].to_numpy()

    train_x = features[train_indices]
    train_y = target[train_indices]
    val_x = features[val_indices]
    val_y = target[val_indices]
    test_x = features[test_indices]
    test_y = target[test_indices]

    return train_x, train_y, val_x, val_y, test_x, test_y


def shuffle_sets(train_x, train_y, val_x, val_y, test_x, test_y):
    """
    Shuffle the data within each of the training, validation, and test sets.

    :param:
    - train_x, train_y: Numpy arrays containing the training features and labels
    - val_x, val_y: Numpy arrays containing the validation features and labels
    - test_x, test_y: Numpy arrays containing the test features and labels

    :return:
    - Shuffled versions of train_x, train_y, val_x, val_y, test_x, test_y
    """
    train_x, train_y = shuffle(train_x, train_y)
    val_x, val_y = shuffle(val_x, val_y)
    test_x, test_y = shuffle(test_x, test_y)

    return train_x, train_y, val_x, val_y, test_x, test_y


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
    tsne = TSNE(n_components=2, random_state=42)
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
    file_path = './cme_and_electron/cme_josias_10MeV.csv'
    df = pd.read_csv(file_path)
    # Split the data into training, validation, and test sets
    # Test the function
    train_x, train_y, val_x, val_y, test_x, test_y = split_data(df)

    shuffled_train_x, shuffled_train_y, shuffled_val_x, shuffled_val_y, shuffled_test_x, shuffled_test_y = shuffle_sets(
        train_x, train_y, val_x, val_y, test_x, test_y)

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
    feature_extractor.load_weights('model_weights_2023-09-28_13-46-32.h5')
    print('weights loaded successfully!')

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # training
    Options = {
        'batch_size': 768,
        'epochs': 1000,
        'patience': 25,
        'learning_rate': 3e-3,
    }

    # print options used
    print(Options)
    mb.train_features(feature_extractor, shuffled_train_x, shuffled_train_y, shuffled_val_x, shuffled_val_y,
                      learning_rate=Options['learning_rate'],
                      epochs=Options['epochs'],
                      batch_size=Options['batch_size'],
                      patience=Options['patience'], save_tag=timestamp)

    plot_tsne_and_save_with_timestamp(feature_extractor, shuffled_train_x, shuffled_train_y, 'training',
                                      save_tag=timestamp)

    plot_tsne_and_save_with_timestamp(feature_extractor, shuffled_test_x, shuffled_test_y, 'testing',
                                      save_tag=timestamp)


if __name__ == '__main__':
    main()
