from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from models import modeling
from evaluate import evaluation as eval
from dataload import seploader as sepl
import tensorflow as tf

SEED = 42  # seed number


def plot_tsne_and_save_extended(model, X, y, title, prefix, sep_shape='o', model_type='features', save_tag=None):
    """
    Applies t-SNE to the features extracted by the given model and saves the plot in 2D with a timestamp.
    The color of the points is determined by their label values.

    Parameters:
    - model: Trained feature extractor model
    - X: Input data (NumPy array or compatible)
    - y: Target labels (NumPy array or compatible)
    - title: Title for the plot
    - prefix: Prefix for the file name
    - sep_shape: The shape of the marker to use for SEP events (above threshold).
    - model_type: The type of model to use (feature, feature_reg, feature_reg_dec)
    - save_tag: Optional tag to add to the saved file name

    Returns:
    - Saves a 2D t-SNE plot to a file with a timestamp
    """
    # Define the threshold
    threshold = np.log(10 / np.exp(2)) + 1e8  # threshold

    # Extract features using the trained extended model
    if model_type == 'features_reg_dec':
        features, _, _ = model.predict(X)
    elif model_type == 'features_reg':
        features, _ = model.predict(X)
    else:  # model_type == 'features'
        features = model.predict(X)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features)

    # Identify above and below threshold indices
    above_threshold_indices = np.where(y > threshold)[0]
    below_threshold_indices = np.where(y <= threshold)[0]

    # Create scatter plot for below-threshold points (in gray)
    plt.figure(figsize=(12, 8))
    plt.scatter(tsne_result[below_threshold_indices, 0], tsne_result[below_threshold_indices, 1],
                color='gray', alpha=0.6, label='Below Threshold')

    # Overlay scatter plot for above-threshold points (in plasma palette)
    scatter = plt.scatter(tsne_result[above_threshold_indices, 0], tsne_result[above_threshold_indices, 1],
                          c=y[above_threshold_indices], cmap='plasma', alpha=1.0, marker=sep_shape,
                          label='Above Threshold')

    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Label Value')

    # Add legend to differentiate above-threshold points
    plt.legend()

    plt.title(f'{title}\n2D t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Save the plot
    file_path = f"{prefix}_tsne_plot_{str(save_tag)}.png"
    plt.savefig(file_path)
    plt.close()


def plot_tsne_and_save_with_timestamp(model, X, y, title, prefix, save_tag=None):
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

    # Create scatter plot for below-threshold points (in gray)
    plt.figure(figsize=(12, 8))
    plt.scatter(tsne_result[below_threshold_indices, 0], tsne_result[below_threshold_indices, 1],
                color='gray', alpha=0.6, label='Below Threshold')

    # Overlay scatter plot for above-threshold points (in plasma palette)
    scatter = plt.scatter(tsne_result[above_threshold_indices, 0], tsne_result[above_threshold_indices, 1],
                          c=y[above_threshold_indices], cmap='plasma', alpha=1.0, edgecolors='r',
                          label='Above Threshold')

    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Label Value')

    # Add legend to differentiate above-threshold points
    plt.legend()

    plt.title(f'{title}\n2D t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Save the plot
    file_path = f"{prefix}_tsne_plot_{str(save_tag)}.png"
    plt.savefig(file_path)
    plt.close()


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


def load_and_plot_tsne(model_path, model_type, title, sep_marker, data_dir='./cme_and_electron/data'):
    """
    Load a trained model and plot its t-SNE visualization.

    :param model_path: Path to the saved model.
    :param model_type: Type of model to load ('features', 'features_reg', 'features_reg_dec').
    :param title: Title for the t-SNE plot.
    :param sep_marker: The shape of the marker to use for SEP events (above threshold).
    :param data_dir: Directory where the data files are stored.
    """

    # print the parameters
    print('Model type:', model_type)
    print('Model path:', model_path)
    print('Title:', title)
    print('SEP marker:', sep_marker)
    print('Data directory:', data_dir)

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # check for gpus
    tf.config.list_physical_devices('GPU')
    # Load the appropriate model
    mb = modeling.ModelBuilder()
    if model_type == 'features':
        loaded_model = mb.create_model_feat(inputs=19, feat_dim=9, hiddens=[18])
    elif model_type == 'features_reg':
        loaded_model = mb.create_model(inputs=19, feat_dim=9, outputs=1, hiddens=[18])
    else:  # features_reg_dec
        loaded_model = mb.create_model_with_ae(inputs=19, feat_dim=9, outputs=1, hiddens=[18])

    loaded_model.load_weights(model_path)
    print(f'Model loaded from {model_path}')

    # Load data
    loader = sepl.SEPLoader()
    train_x, train_y, val_x, val_y, test_x, test_y = loader.load_from_dir(data_dir)
    # Extract counts of events
    train_count = count_above_threshold(train_y)
    val_count = count_above_threshold(val_y)
    test_count = count_above_threshold(test_y)

    print(f'Training set: {train_count} above the threshold')
    print(f'Validation set: {val_count} above the threshold')
    print(f'Test set: {test_count} above the threshold')

    # Combine training and validation sets
    combined_train_x, combined_train_y = loader.combine(train_x, train_y, val_x, val_y)

    # Plot and save t-SNE
    plot_tsne_and_save_extended(loaded_model,
                                test_x, test_y,
                                title,
                                model_type + '_testing_',
                                model_type=model_type,
                                save_tag=timestamp)
    plot_tsne_and_save_extended(loaded_model,
                                combined_train_x,
                                combined_train_y,
                                title,
                                model_type + '_training_',
                                model_type=model_type,
                                save_tag=timestamp)



def load_and_test(model_path, model_type, title, threshold=10, data_dir='./cme_and_electron/data'):
    """
    Load a trained model and evaluate its performance.

    :param model_path: Path to the saved model.
    :param model_type: Type of model to load ('features', 'features_reg', 'features_reg_dec').
    :param title: Title for the evaluation results.
    :param threshold: Threshold for evaluation.
    :param data_dir: Directory where the data files are stored.
    """

    # print the parameters
    print('Model type:', model_type)
    print('Model path:', model_path)
    print('Title:', title)
    print('Threshold:', threshold)
    print('Data directory:', data_dir)

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # check for gpus
    tf.config.list_physical_devices('GPU')
    # Load the appropriate model
    mb = modeling.ModelBuilder()
    if model_type == 'features_reg_dec' or model_type == 'features_reg':
        loaded_model = mb.create_model(inputs=19, feat_dim=9, outputs=1, hiddens=[18])
    else:  # features
        feature_extractor = mb.create_model_feat(inputs=19, feat_dim=9, hiddens=[18])
        loaded_model = mb.add_regression_head_with_proj(feature_extractor, freeze_features=False)

    loaded_model.load_weights(model_path)
    print(f'Model loaded from {model_path}')

    # Load data
    loader = sepl.SEPLoader()
    train_x, train_y, val_x, val_y, test_x, test_y = loader.load_from_dir(data_dir)
    # Extract counts of events
    train_count = count_above_threshold(train_y)
    val_count = count_above_threshold(val_y)
    test_count = count_above_threshold(test_y)

    print(f'Training set: {train_count} above the threshold')
    print(f'Validation set: {val_count} above the threshold')
    print(f'Test set: {test_count} above the threshold')

    # Combine training and validation sets
    combined_train_x, combined_train_y = loader.combine(train_x, train_y, val_x, val_y)

    # Evaluate and save results
    ev = eval.Evaluator()
    ev.evaluate(loaded_model, test_x, test_y, title, threshold=threshold, save_tag=model_type + '_test_' + timestamp)
    ev.evaluate(loaded_model, combined_train_x, combined_train_y, title, threshold=threshold,
                save_tag=model_type + '_training_' + timestamp)
