import numpy as np
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
