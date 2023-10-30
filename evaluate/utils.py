from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from models import modeling
from evaluate import evaluation as eval
from dataload import seploader as sepl
import tensorflow as tf
from typing import List, Tuple

SEED = 42  # seed number


def plot_tsne_extended(model, X, y, title, prefix, model_type='features_reg', save_tag=None):
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
    - model_type: The type of model to use (feature, feature_reg, features_reg_dec)
    - save_tag: Optional tag to add to the saved file name

    Returns:
    - Saves a 2D t-SNE plot to a file with a timestamp
    """
    # Define the thresholds
    threshold = np.log(10 / np.exp(2)) + 1e-4
    sep_threshold = np.log(10)

    # Extract features using the trained extended model
    if model_type == 'features_reg_dec':
        features, _, _ = model.predict(X)
    else:  # model_type == 'features' or 'features_reg'
        features, _ = model.predict(X)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features)

    # Identify indices based on thresholds
    above_sep_threshold_indices = np.where(y > sep_threshold)[0]
    elevated_event_indices = np.where((y > threshold) & (y <= sep_threshold))[0]
    below_threshold_indices = np.where(y <= threshold)[0]

    plt.figure(figsize=(12, 8))

    # Create scatter plot for below-threshold points (in gray)
    plt.scatter(tsne_result[below_threshold_indices, 0], tsne_result[below_threshold_indices, 1], marker='o',
                color='gray', alpha=0.6)

    # Normalize y-values for better color mapping
    norm = plt.Normalize(y.min(), y.max())

    # Compute marker sizes based on y-values. Squaring to amplify differences.
    marker_sizes_elevated = 50 * ((y[elevated_event_indices] - y.min()) / (y.max() - y.min())) ** 2 + 10
    marker_sizes_sep = 50 * ((y[above_sep_threshold_indices] - y.min()) / (y.max() - y.min())) ** 2 + 10

    # Create scatter plot for elevated events (square marker)
    plt.scatter(tsne_result[elevated_event_indices, 0], tsne_result[elevated_event_indices, 1],
                c=y[elevated_event_indices], cmap='plasma', norm=norm, alpha=0.6, marker='o', s=marker_sizes_elevated)

    # Create scatter plot for SEPs (diamond marker)
    plt.scatter(tsne_result[above_sep_threshold_indices, 0], tsne_result[above_sep_threshold_indices, 1],
                c=y[above_sep_threshold_indices], cmap='plasma', norm=norm, alpha=0.6, marker='d', s=marker_sizes_sep)

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('ln Intensity')

    # Add a legend
    legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10),
                     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
                     plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='red', markersize=10)]

    plt.legend(legend_labels, ['Background', 'Elevated Events (darker colors)', 'SEPs (lighter colors)'],
               loc='upper left')

    plt.title(f'{title}\n2D t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Save the plot
    file_path = f"{prefix}_tsne_plot_{str(save_tag)}.png"
    plt.savefig(file_path)
    plt.close()

    return file_path


def plot_tsne_pds(model, X, y, title, prefix, save_tag=None):
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
    # Define the thresholds
    threshold = np.log(10 / np.exp(2)) + 1e-4
    sep_threshold = np.log(10)

    # Extract features using the trained model
    features = model.predict(X)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=SEED)
    tsne_result = tsne.fit_transform(features)

    # Identify indices based on thresholds
    above_sep_threshold_indices = np.where(y > sep_threshold)[0]
    elevated_event_indices = np.where((y > threshold) & (y <= sep_threshold))[0]
    below_threshold_indices = np.where(y <= threshold)[0]

    plt.figure(figsize=(12, 8))

    # Create scatter plot for below-threshold points (in gray)
    plt.scatter(tsne_result[below_threshold_indices, 0], tsne_result[below_threshold_indices, 1], marker='o',
                color='gray', alpha=0.6)

    # Normalize y-values for better color mapping
    norm = plt.Normalize(y.min(), y.max())

    # Compute marker sizes based on y-values
    marker_sizes_elevated = 50 * ((y[elevated_event_indices] - y.min()) / (y.max() - y.min())) ** 2 + 10
    marker_sizes_sep = 50 * ((y[above_sep_threshold_indices] - y.min()) / (y.max() - y.min())) ** 2 + 10

    # Create scatter plot for elevated events (square marker)
    plt.scatter(tsne_result[elevated_event_indices, 0], tsne_result[elevated_event_indices, 1],
                c=y[elevated_event_indices], cmap='plasma', norm=norm, alpha=0.6, marker='o', s=marker_sizes_elevated)

    # Create scatter plot for SEPs (diamond marker)
    plt.scatter(tsne_result[above_sep_threshold_indices, 0], tsne_result[above_sep_threshold_indices, 1],
                c=y[above_sep_threshold_indices], cmap='plasma', norm=norm, alpha=0.6, marker='d', s=marker_sizes_sep)

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Label Value')

    # Add legend
    # Add a legend
    legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10),
                     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
                     plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='red', markersize=10)]

    plt.legend(legend_labels, ['Background', 'Elevated Events (darker colors)', 'SEPs (lighter colors)'],
               loc='upper left')

    plt.title(f'{title}\n2D t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Save the plot
    file_path = f"{prefix}_tsne_plot_{str(save_tag)}.png"
    plt.savefig(file_path)
    plt.close()

    return file_path


def count_above_threshold(y_values: List[float], threshold: float = 0.3027, sep_threshold: float = 2.3026) -> Tuple[
    int, int]:
    """
    Count the number of y values that are above a given threshold for elevated events and above a sep_threshold for sep events.

    Parameters:
    - y_values (List[float]): The array of y-values to check.
    - threshold (float, optional): The threshold value to use for counting elevated events. Default is log(10 / e^2).
    - sep_threshold (float, optional): The threshold value to use for counting sep events. Default is log(10).

    Returns:
    - Tuple[int, int]: The count of y-values that are above the thresholds for elevated and sep events, respectively.
    """

    y_values_np = np.array(y_values)  # Convert list to NumPy array
    elevated_count = np.sum((y_values_np > threshold) & (y_values_np <= sep_threshold))
    sep_count = np.sum(y_values_np > sep_threshold)

    return elevated_count, sep_count


def load_and_plot_tsne(model_path, model_type, title, sep_marker, data_dir='/home1/jmoukpe2016/keras-functional-api/cme_and_electron/data'):
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
    print(tf.config.list_physical_devices('GPU'))
    # Load the appropriate model
    mb = modeling.ModelBuilder()
    if model_type == 'features':
        features_model = mb.create_model_pds(input_dim=19, feat_dim=9, hiddens=[18])
        loaded_model = mb.add_reg_proj_head(features_model, freeze_features=False, pds=True)
    elif model_type == 'features_reg':
        features_model = mb.create_model(input_dim=19, feat_dim=9, output_dim=1, hiddens=[18])
        loaded_model = mb.add_reg_proj_head(features_model, freeze_features=False)
    elif model_type == 'features_reg_dec':
        features_model = mb.create_model(input_dim=19, feat_dim=9, output_dim=1, hiddens=[18], with_ae=True)
        loaded_model = mb.add_reg_proj_head(features_model, freeze_features=False)
    else:  # regular reg
        loaded_model = mb.create_model(input_dim=19, feat_dim=9, output_dim=1, hiddens=[18])

    loaded_model.load_weights(model_path)
    print(f'Model loaded from {model_path}')

    # Load data
    loader = sepl.SEPLoader()
    train_x, train_y, val_x, val_y, test_x, test_y = loader.load_from_dir(data_dir)
    # Extract counts of events
    elevateds, seps = count_above_threshold(train_y)
    print(f'Sub-Training set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(val_y)
    print(f'Validation set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(test_y)
    print(f'Test set: elevated events: {elevateds}  and sep events: {seps}')

    # Combine training and validation sets
    combined_train_x, combined_train_y = loader.combine(train_x, train_y, val_x, val_y)

    # Plot and save t-SNE
    plot_tsne_extended(loaded_model,
                       test_x, test_y,
                       title,
                       model_type + '_testing_',
                       model_type=model_type,
                       save_tag=timestamp)
    plot_tsne_extended(loaded_model,
                       combined_train_x,
                       combined_train_y,
                       title,
                       model_type + '_training_',
                       model_type=model_type,
                       save_tag=timestamp)


def load_and_test(model_path, model_type, title, threshold=10, data_dir='/home1/jmoukpe2016/keras-functional-api/cme_and_electron/data'):
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
    print(tf.config.list_physical_devices('GPU'))
    # Load the appropriate model
    mb = modeling.ModelBuilder()

    if model_type == 'features':
        features_model = mb.create_model_pds(input_dim=19, feat_dim=9, hiddens=[18])
        loaded_model = mb.add_reg_proj_head(features_model, freeze_features=False)
    elif model_type == 'features_reg':
        features_model = mb.create_model(input_dim=19, feat_dim=9, output_dim=1, hiddens=[18])
        loaded_model = mb.add_reg_proj_head(features_model, freeze_features=False)
    else:  # features_reg_dec
        features_model = mb.create_model(input_dim=19, feat_dim=9, output_dim=1, hiddens=[18], with_ae=True)
        loaded_model = mb.add_reg_proj_head(features_model, freeze_features=False)

    loaded_model.load_weights(model_path)
    print(f'Model loaded from {model_path}')

    # Load data
    loader = sepl.SEPLoader()
    train_x, train_y, val_x, val_y, test_x, test_y = loader.load_from_dir(data_dir)
    # Extract counts of events
    elevateds, seps = count_above_threshold(train_y)
    print(f'Sub-Training set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(val_y)
    print(f'Validation set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(test_y)
    print(f'Test set: elevated events: {elevateds}  and sep events: {seps}')

    # Combine training and validation sets
    combined_train_x, combined_train_y = loader.combine(train_x, train_y, val_x, val_y)

    # Evaluate and save results
    ev = eval.Evaluator()
    ev.evaluate(loaded_model, test_x, test_y, title, threshold=threshold, save_tag=model_type + '_test_' + timestamp)
    ev.evaluate(loaded_model, combined_train_x, combined_train_y, title, threshold=threshold,
                save_tag=model_type + '_training_' + timestamp)
