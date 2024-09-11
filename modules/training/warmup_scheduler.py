import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


class WarmupScheduler(Callback):
    """
    Custom callback to implement learning rate warmup. Gradually increases the learning rate from
    initial_lr to target_lr over a specified number of warmup steps.

    Attributes:
        warmup_steps (int): Number of steps (batches) over which the learning rate increases.
        initial_lr (float): Starting learning rate for the warmup.
        target_lr (float): Target learning rate after the warmup is complete.
    """

    def __init__(self, warmup_steps: int, initial_lr: float, target_lr: float) -> None:
        """
        Initializes the WarmupScheduler callback.

        Args:
            warmup_steps (int): The number of steps over which the learning rate increases.
            initial_lr (float): The initial learning rate to start with.
            target_lr (float): The final learning rate after warmup is done.
        """
        super(WarmupScheduler, self).__init__()
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.target_lr = target_lr

    def on_train_begin(self, logs: dict = None) -> None:
        """
        Sets the learning rate to the initial learning rate at the start of training.

        Args:
            logs (dict, optional): A dictionary of logs from Keras. Defaults to None.
        """
        K.set_value(self.model.optimizer.lr, self.initial_lr)

    def on_batch_end(self, batch: int, logs: dict = None) -> None:
        """
        Adjusts the learning rate after each batch during the warmup phase.

        Args:
            batch (int): The current batch number.
            logs (dict, optional): A dictionary of logs from Keras. Defaults to None.
        """
        if batch < self.warmup_steps:
            # Linearly increase learning rate from initial_lr to target_lr
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (batch / self.warmup_steps)
            K.set_value(self.model.optimizer.lr, lr)
        else:
            # Once warmup is complete, set learning rate to target_lr
            K.set_value(self.model.optimizer.lr, self.target_lr)
