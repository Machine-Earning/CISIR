from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

SEED = 42  # seed number


def plot_tsne_and_save_extended(model, X, y, title, prefix, with_ae=False, save_tag=None):
    """
    Applies t-SNE to the features extracted by the given extended model and saves the plot in 2D with a timestamp.
    The color of the points is determined by their label values.

    Parameters:
    - model: Trained extended feature extractor model
    - X: Input data (NumPy array or compatible)
    - y: Target labels (NumPy array or compatible)
    - prefix: Prefix for the file name

    Returns:
    - Saves a 2D t-SNE plot to a file with a timestamp
    """
    # Define the threshold
    threshold = np.log(10)

    # Extract features using the trained extended model
    if with_ae:
        features, _, _ = model.predict(X)
    else:
        features, _ = model.predict(X)

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