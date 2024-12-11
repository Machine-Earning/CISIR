import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import Callback


class SmoothEarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving, with optional smoothing.

    This callback monitors a quantity and stops training when it stops improving,
    with options to apply smoothing to the monitored metric before making decisions.

    Parameters:
    -----------
    monitor : str
        Quantity to be monitored (e.g., 'val_loss', 'val_accuracy').
    min_delta : float, optional
        Minimum change in the monitored quantity to qualify as an improvement.
        Defaults to 0.
    patience : int
        Number of epochs with no improvement after which training will be stopped.
    mode : {'auto', 'min', 'max'}, optional
        Mode to determine if an improvement has occurred. In 'min' mode, training
        will stop when the quantity monitored has stopped decreasing; in 'max' mode
        it will stop when the quantity monitored has stopped increasing; in 'auto'
        mode, the direction is automatically inferred from the name of the monitored
        quantity.
    restore_best_weights : bool, optional
        Whether to restore model weights from the epoch with the best value of
        the monitored quantity. If False, the model weights obtained at the last
        step of training are used. Defaults to False.
    smoothing_method : {'none', 'moving_average', 'exponential_moving_average', 'gaussian'}, optional
        The smoothing method to apply to the monitored metric.
        - 'none': No smoothing applied.
        - 'moving_average': Simple moving average over a window.
        - 'exponential_moving_average': Exponential moving average with decay.
        - 'gaussian': Gaussian-weighted moving average over a window.
    smoothing_parameters : dict, optional
        Parameters specific to the chosen smoothing method.
        - For 'moving_average':
            - 'window_size': int, number of recent epochs to consider.
        - For 'exponential_moving_average':
            - 'alpha': float, smoothing factor between 0 and 1.
        - For 'gaussian':
            - 'window_size': int, number of recent epochs to consider.
            - 'stdev': float, standard deviation of the Gaussian kernel.
    verbose : int, optional
        Verbosity mode. 0 = silent, 1 = messages at each epoch. Defaults to 0.

    Notes:
    ------
    - The smoothed metric is used to determine improvements.
    - The patience parameter counts epochs based on the smoothed metric.
    """

    def __init__(self,
                 monitor: str = 'val_loss',
                 min_delta: float = 0,
                 patience: int = 0,
                 mode: str = 'auto',
                 restore_best_weights: bool = False,
                 smoothing_method: str = 'none',
                 smoothing_parameters: dict = None,
                 verbose: int = 0):
        super(SmoothEarlyStopping, self).__init__()

        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.smoothing_method = smoothing_method
        if smoothing_parameters is None:
            smoothing_parameters = {}
        self.smoothing_parameters = smoothing_parameters

        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        # List to store metric history
        self.metric_history = []

        # Determine the direction for the monitored quantity
        if mode not in ['auto', 'min', 'max']:
            print(f'SmoothEarlyStopping mode {mode} is unknown, fallback to auto mode.')
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
            self.best = -np.Inf
        else:
            if self.monitor.endswith('acc') or self.monitor.endswith('accuracy') or self.monitor.endswith('pcc'):
                self.monitor_op = np.greater
                self.min_delta *= 1
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.min_delta *= -1
                self.best = np.Inf

        # Initialize smoothed metric
        self.best_smoothed_metric = self.best

        # For exponential moving average
        self.previous_smoothed_metric = None

        # For Gaussian smoothing
        if self.smoothing_method == 'gaussian':
            self.gaussian_weights = self._compute_gaussian_weights()

    def _compute_gaussian_weights(self) -> np.ndarray:
        """
        Compute Gaussian weights for the Gaussian smoothing method.

        Returns:
        --------
        weights : np.ndarray
            Array of Gaussian weights normalized to sum to 1.
        """
        window_size = self.smoothing_parameters.get('window_size', 5)
        stdev = self.smoothing_parameters.get('stdev', 1.0)

        # Ensure window_size is at least 1
        window_size = max(int(window_size), 1)

        # Create an array of indices centered at the middle
        indices = np.arange(window_size) - (window_size - 1) / 2.0
        # Compute Gaussian weights
        weights = np.exp(-0.5 * (indices / stdev) ** 2)
        # Normalize weights to sum to 1
        weights /= np.sum(weights)

        return weights

    def compute_smoothed_metric(self) -> float:
        """
        Compute the smoothed metric based on the chosen smoothing method.

        Returns:
        --------
        smoothed_metric : float
            The smoothed metric value.
        """
        if self.smoothing_method == 'none':
            # No smoothing, return the last metric value
            smoothed_metric = self.metric_history[-1]
        elif self.smoothing_method == 'moving_average':
            window_size = self.smoothing_parameters.get('window_size', 5)
            # Use available values if not enough history, TODO: Check if this is correct
            metrics = self.metric_history[-window_size:]
            smoothed_metric = np.mean(metrics)
        elif self.smoothing_method == 'exponential_moving_average':
            alpha = self.smoothing_parameters.get('alpha', 0.3)
            if self.previous_smoothed_metric is None:
                # Initialize with the first metric value
                smoothed_metric = self.metric_history[-1]
            else:
                smoothed_metric = alpha * self.metric_history[-1] + (1 - alpha) * self.previous_smoothed_metric
            # Update the previous smoothed metric
            self.previous_smoothed_metric = smoothed_metric
        elif self.smoothing_method == 'gaussian':
            window_size = self.smoothing_parameters.get('window_size', 5)
            weights = self.gaussian_weights
            # Use available values if not enough history
            metrics = self.metric_history[-window_size:]
            # Adjust weights if not enough metrics
            if len(metrics) < len(weights):
                weights = weights[-len(metrics):]
                weights /= np.sum(weights)
            smoothed_metric = np.sum(weights * metrics)
        else:
            raise ValueError(f"Unknown smoothing method: {self.smoothing_method}")

        return smoothed_metric

    def on_train_begin(self, logs=None):
        """Reset variables at the beginning of training."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_smoothed_metric = self.best
        self.metric_history = []
        self.previous_smoothed_metric = None

        # Recompute Gaussian weights if necessary
        if self.smoothing_method == 'gaussian':
            self.gaussian_weights = self._compute_gaussian_weights()

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """
        Check at the end of each epoch whether training should stop.

        Parameters:
        -----------
        epoch : int
            Current epoch index.
        logs : dict
            Dictionary of logs from the epoch.
        """
        current = logs.get(self.monitor)
        if current is None:
            print(f"SmoothEarlyStopping: Monitored metric '{self.monitor}' is not available.")
            return

        self.metric_history.append(current)
        smoothed_metric = self.compute_smoothed_metric()

        if self.monitor_op(smoothed_metric - self.min_delta, self.best_smoothed_metric):
            # Improvement found
            self.best_smoothed_metric = smoothed_metric
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            # No improvement
            self.wait += 1
            if self.wait >= self.patience:
                # Patience exceeded, stop training
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        print(
                            f"Restoring model weights from the end of the best epoch: {self.stopped_epoch - self.patience + 1}")
                    self.model.set_weights(self.best_weights)

        if self.verbose > 0:
            print(f"Epoch {epoch + 1}: smoothed {self.monitor} = {smoothed_metric:.6f}")

    def on_train_end(self, logs=None):
        """Log training end information."""
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")


def find_optimal_epoch_by_smoothing(
        metric_history,
        smoothing_method='none',
        smoothing_parameters=None,
        mode='auto',
        plot=False):
    """
    Smooths the metric history using the specified smoothing method and finds the epoch corresponding
    to the best (minimum or maximum) smoothed metric value.

    Parameters:
    - metric_history: List or array of metric values over epochs.
    - smoothing_method: {'none', 'moving_average', 'exponential_moving_average', 'gaussian'}, optional
        The smoothing method to apply.
        - 'none': No smoothing applied.
        - 'moving_average': Simple moving average over a centered window.
        - 'exponential_moving_average': Exponential moving average with symmetric window.
        - 'gaussian': Gaussian-weighted moving average over a centered window.
    - smoothing_parameters: dict, optional
        Parameters specific to the chosen smoothing method.
        - For 'moving_average':
            - 'window_size': int, number of epochs to consider (must be odd for symmetry).
        - For 'exponential_moving_average':
            - 'alpha': float, smoothing factor between 0 and 1.
            - 'window_size': int, number of epochs to consider (must be odd for symmetry).
        - For 'gaussian':
            - 'window_size': int, number of epochs to consider (must be odd for symmetry).
            - 'stdev': float, standard deviation of the Gaussian kernel.
    - mode: {'auto', 'min', 'max'}, optional
        Mode to determine if the optimal metric is a minimum or maximum.
        - 'min': Look for minimum metric value.
        - 'max': Look for maximum metric value.
        - 'auto': Infer from common metric names ('loss' -> 'min', 'acc'/'accuracy' -> 'max').
    - plot: Boolean flag to indicate if the plot should be generated (default: False).

    Returns:
    - optimal_epoch: Epoch corresponding to the best smoothed metric (1-based index).
    """

    if smoothing_parameters is None:
        smoothing_parameters = {}

    metric_history = np.array(metric_history)
    epochs = np.arange(len(metric_history)) + 1  # Epochs start from 1

    # Determine mode if 'auto'
    if mode == 'auto':
        # Since we don't have the metric name, default to 'min'
        mode = 'min'

    if mode == 'min':
        monitor_op = np.less
        best = np.Inf
    elif mode == 'max':
        monitor_op = np.greater
        best = -np.Inf
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Compute smoothed metric history
    if smoothing_method == 'none':
        smoothed_metrics = metric_history.copy()
    elif smoothing_method == 'moving_average':
        window_size = smoothing_parameters.get('window_size', 5)
        smoothed_metrics = centered_moving_average(metric_history, window_size)
    elif smoothing_method == 'exponential_moving_average':
        window_size = smoothing_parameters.get('window_size', 5)
        alpha = smoothing_parameters.get('alpha', 0.3)
        smoothed_metrics = centered_exponential_moving_average(metric_history, window_size, alpha)
    elif smoothing_method == 'gaussian':
        window_size = smoothing_parameters.get('window_size', 5)
        stdev = smoothing_parameters.get('stdev', 1.0)
        smoothed_metrics = centered_gaussian_smoothing(metric_history, window_size, stdev)
    else:
        raise ValueError(f"Unknown smoothing method: {smoothing_method}")

    # Find the optimal epoch
    if mode == 'min':
        optimal_index = np.nanargmin(smoothed_metrics)
    else:
        optimal_index = np.nanargmax(smoothed_metrics)
    optimal_epoch = epochs[optimal_index]

    # Plotting (if requested)
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, metric_history, 'bo-', label='Original Metric History')
        plt.plot(epochs, smoothed_metrics, 'r--', label=f'Smoothed Metric ({smoothing_method})')
        plt.axvline(optimal_epoch, color='g', linestyle=':', label=f'Optimal Epoch ({optimal_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Metric History and Smoothed Metric')
        plt.legend()
        plt.grid(True)
        plt.show()

    return optimal_epoch


def centered_moving_average(data, window_size):
    """
    Computes the centered moving average of the data.

    Parameters:
    - data: 1D array of data points.
    - window_size: Size of the moving window (must be odd for symmetry).

    Returns:
    - smoothed_data: 1D array of smoothed data.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd for centered moving average.")
    k = (window_size - 1) // 2
    padded_data = np.pad(data, (k, k), mode='edge')
    smoothed_data = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data


def centered_exponential_moving_average(data, window_size, alpha):
    """
    Computes the centered exponential moving average of the data.

    Parameters:
    - data: 1D array of data points.
    - window_size: Size of the window (must be odd for symmetry).
    - alpha: Smoothing factor between 0 and 1.

    Returns:
    - smoothed_data: 1D array of smoothed data.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd for centered exponential moving average.")
    k = (window_size - 1) // 2
    weights = [alpha * (1 - alpha) ** abs(i) for i in range(-k, k + 1)]
    weights = np.array(weights)
    weights /= weights.sum()
    padded_data = np.pad(data, (k, k), mode='edge')
    smoothed_data = np.convolve(padded_data, weights, mode='valid')
    return smoothed_data


def centered_gaussian_smoothing(data, window_size, stdev):
    """
    Computes the centered Gaussian smoothing of the data.

    Parameters:
    - data: 1D array of data points.
    - window_size: Size of the window (must be odd for symmetry).
    - stdev: Standard deviation of the Gaussian kernel.

    Returns:
    - smoothed_data: 1D array of smoothed data.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd for centered Gaussian smoothing.")
    k = (window_size - 1) // 2
    indices = np.arange(-k, k + 1)
    weights = np.exp(-0.5 * (indices / stdev) ** 2)
    weights /= weights.sum()
    padded_data = np.pad(data, (k, k), mode='edge')
    smoothed_data = np.convolve(padded_data, weights, mode='valid')
    return smoothed_data


def find_optimal_epoch_by_quadratic_fit(metric_history, plot=False):
    """
    Fits a quadratic function to the metric history and finds the epoch corresponding to the minimum of the quadratic fit.
    Optionally plots the original metric history and the fitted quadratic curve.

    Parameters:
    - metric_history: List or array of metric values over epochs.
    - plot: Boolean flag to indicate if the plot should be generated (default: False).

    Returns:
    - optimal_epoch: Epoch corresponding to the minimum of the quadratic fit (may be fractional).
    """

    # Epochs start from 1
    epochs = np.arange(len(metric_history)) + 1
    y = np.array(metric_history)

    # Fit quadratic: y = a*x^2 + b*x + c
    coefficients = np.polyfit(epochs, y, deg=2)
    a, b, c = coefficients

    # Calculate the fitted quadratic values
    quadratic_fit = a * epochs ** 2 + b * epochs + c

    # Handle the case where 'a' is zero (linear function)
    if a == 0:
        optimal_epoch = epochs[np.argmin(y)]
    else:
        optimal_epoch = -b / (2 * a)  # Keep this fractional for better accuracy
        # Ensure the optimal epoch is within the valid range
        optimal_epoch = max(min(optimal_epoch, epochs[-1]), epochs[0])

    # Plotting (if requested)
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, y, 'bo-', label='Original Metric History')
        plt.plot(epochs, quadratic_fit, 'r--', label='Quadratic Fit')
        plt.axvline(optimal_epoch, color='g', linestyle=':', label=f'Optimal Epoch ({optimal_epoch:.2f})')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Metric History and Quadratic Fit')
        plt.legend()
        plt.grid(True)
        plt.show()

    return optimal_epoch
