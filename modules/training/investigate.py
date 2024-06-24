# types for type hinting
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
# imports
from numpy import ndarray
from tensorflow.keras import callbacks, Model
from cme_modeling import ModelBuilder, evaluate


class InvestigateCallback(callbacks.Callback):
    """
    Custom callback to evaluate the model on SEP samples at the end of each epoch.
    """

    def __init__(self,
                 model: Model,
                 X_train: ndarray,
                 y_train: ndarray,
                 batch_size: int,
                 model_builder: ModelBuilder,
                 save_tag: Optional[str] = None):
        super().__init__()
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size if batch_size > 0 else None
        self.sep_threshold = np.log(10)
        self.threshold = np.log(10.0 / np.exp(2))
        self.save_tag = save_tag
        self.sep_sep_losses = []
        # self.losses = []
        # self.epochs_10s = []
        self.model_builder = model_builder
        self.sep_sep_count = 0
        self.sep_sep_counts = []
        self.cumulative_sep_sep_count = 0  # Initialize cumulative count
        self.cumulative_sep_sep_counts = []
        self.total_counts = []
        self.batch_counts = []
        self.sep_sep_percentages = []
        # the losses
        self.pair_type_losses = {  # Store losses for each pair type
            'sep_sep': [],
            'sep_elevated': [],
            'sep_background': [],
            'elevated_elevated': [],
            'elevated_background': [],
            'background_background': []
        }
        self.overall_losses = []  # Store overall losses

    def on_batch_end(self, batch, logs=None):
        """
        Actions to be taken at the end of each batch.

        :param batch: the index of the batch within the current epoch.
        :param logs: the logs containing the metrics results.
        """
        sep_indices = self.find_sep_samples(self.y_train)
        if len(sep_indices) > 0:
            X_sep = self.X_train[sep_indices]
            y_sep = self.y_train[sep_indices]
            # Evaluate the model on SEP samples
            # sep_sep_loss = self.model.evaluate(X_sep, y_sep, batch_size=len(self.y_train), verbose=0)
            sep_sep_loss = evaluate(self.model, X_sep, y_sep)
            self.sep_sep_losses.append(sep_sep_loss)

        # Add the SEP-SEP count for the current batch to the cumulative count
        batch_sep_sep_count = int(self.model_builder.sep_sep_count.numpy())
        print(f'end of batch: {batch}, sep_sep_count: {batch_sep_sep_count} in')
        self.sep_sep_count += batch_sep_sep_count
        self.cumulative_sep_sep_count += batch_sep_sep_count
        self.cumulative_sep_sep_counts.append(self.cumulative_sep_sep_count)
        # Reset for next batch
        self.model_builder.sep_sep_count.assign(0)

    def on_epoch_begin(self, epoch, logs=None):
        """
        Actions to be taken at the beginning of each epoch.

        :param epoch: the index of the epoch.
        :param logs: the logs containing the metrics results.
        """
        # Resetting the counts
        self.sep_sep_count = 0
        self.cumulative_sep_sep_count = 0
        self.model_builder.sep_sep_count.assign(0)
        self.model_builder.sep_elevated_count.assign(0)
        self.model_builder.sep_background_count.assign(0)
        self.model_builder.elevated_elevated_count.assign(0)
        self.model_builder.elevated_background_count.assign(0)
        self.model_builder.background_background_count.assign(0)
        self.model_builder.number_of_batches = 0

    def on_epoch_end(self, epoch, logs=None):
        # Find SEP samples
        # sep_indices = self.find_sep_samples(self.y_train, self.sep_threshold)
        # if len(sep_indices) > 0:
        #     X_sep = self.X_train[sep_indices]
        #     y_sep = self.y_train[sep_indices]
        #     # Evaluate the model on SEP samples
        #     sep_sep_loss = self.model.evaluate(X_sep, y_sep, verbose=0)
        #     self.sep_sep_losses.append(sep_sep_loss)
        #     print(f" Epoch {epoch + 1}: SEP-SEP Loss: {sep_sep_loss}")

        self.collect_losses(epoch)

        # Save the current counts
        self.sep_sep_counts.append(self.sep_sep_count)
        total_count = (
                self.sep_sep_count +
                int(self.model_builder.sep_elevated_count.numpy()) +
                int(self.model_builder.sep_background_count.numpy()) +
                int(self.model_builder.elevated_elevated_count.numpy()) +
                int(self.model_builder.elevated_background_count.numpy()) +
                int(self.model_builder.background_background_count.numpy())
        )
        self.total_counts.append(total_count)
        self.batch_counts.append(self.model_builder.number_of_batches)

        # Calculate and save the percentage of SEP-SEP pairs
        if total_count > 0:
            self.sep_sep_percentages.append((self.sep_sep_count / total_count) * 100)
        else:
            self.sep_sep_percentages.append(0)

        # Reset the counts for the next epoch
        # self.model_builder.sep_sep_count.assign(0)
        self.model_builder.sep_elevated_count.assign(0)
        self.model_builder.sep_background_count.assign(0)
        self.model_builder.elevated_elevated_count.assign(0)
        self.model_builder.elevated_background_count.assign(0)
        self.model_builder.background_background_count.assign(0)
        self.sep_sep_count = 0
        self.model_builder.number_of_batches = 0

        # if epoch % 10 == 9:  # every 10th epoch (considering the first epoch is 0)
        #     loss = self.model.evaluate(self.X_train, self.y_train, batch_size=len(self.y_train), verbose=0)
        #     self.losses.append(loss)
        #     self.epochs_10s.append(epoch + 1)

    def collect_losses(self, epoch):
        """
        Collects and stores the losses for each pair type and overall loss for the given epoch.

        :param epoch: Current epoch number.
        """
        # Evaluate the model and get losses for each pair type, including overall
        pair_losses = evaluate(self.model, self.X_train, self.y_train, pairs=True)

        # Store and print pair type losses
        for pair_type, loss in pair_losses.items():
            if pair_type != 'overall':  # Exclude overall loss here
                self.pair_type_losses[pair_type].append(loss)
                print(f"Epoch {epoch + 1}, {pair_type} Loss: {loss}")

        # Store and print overall loss
        overall_loss = pair_losses['overall']
        self.overall_losses.append(overall_loss)
        print(f"Epoch {epoch + 1}, Overall Loss: {overall_loss}")

    def on_train_end(self, logs=None):
        # At the end of training, save the loss plot
        # self.save_loss_plot()
        # self._save_plot()
        self.save_percent_plot()
        self.save_sep_sep_loss_vs_frequency()
        self.save_slope_of_loss_vs_frequency()
        self.save_combined_loss_plot()

    def find_sep_samples(self, y_train: ndarray) -> ndarray:
        """
        Identifies the indices of SEP samples in the training labels.

        :param y_train: The array of training labels.
        :return: The indices of SEP samples.
        """
        is_sep = y_train > self.sep_threshold
        return np.where(is_sep)[0]

    def find_sample_indices(self, y_train: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Identifies the indices of SEP, elevated, and background samples in the training labels.

        :param y_train: The array of training labels.
        :return: Three arrays containing the indices of SEP, elevated, and background samples respectively.
        """
        is_sep = y_train > self.sep_threshold
        is_elevated = (y_train > self.threshold) & (y_train <= self.sep_threshold)
        is_background = y_train <= self.threshold

        sep_indices = np.where(is_sep)[0]
        elevated_indices = np.where(is_elevated)[0]
        background_indices = np.where(is_background)[0]

        return sep_indices, elevated_indices, background_indices

    # def save_combined_loss_plot(self): """ Saves a combined plot of the losses for each pair type and the overall
    # loss. """ epochs = range(1, len(self.overall_losses) + 1) plt.figure() colors = ['blue', 'green', 'red',
    # 'cyan', 'magenta', 'yellow', 'black']  # Different colors for different curves pair_types = list(
    # self.pair_type_losses.keys()) + ['overall']
    #
    #     for i, pair_type in enumerate(pair_types):
    #         if pair_type == 'overall':
    #             losses = self.overall_losses
    #         else:
    #             losses = self.pair_type_losses[pair_type]
    #         plt.plot(epochs, losses, '-o', label=f'{pair_type} Loss', color=colors[i], markersize=3)
    #
    #     plt.title(f'Losses per Pair Type and Overall Loss Per Epoch, Batch Size {self.batch_size}')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.grid(True)
    #
    #     if self.save_tag:
    #         file_path = f"./investigation/combined_loss_plot_{self.save_tag}.png"
    #     else:
    #         file_path = "./investigation/combined_loss_plot.png"
    #     plt.savefig(file_path)
    #     plt.close()
    #     print(f"Saved combined loss plot at {file_path}")

    def save_combined_loss_plot(self):
        """
        Saves a combined plot of the losses for each pair type and the overall loss as separate subplots.
        """
        epochs = range(1, len(self.overall_losses) + 1)
        pair_types = list(self.pair_type_losses.keys()) + ['overall']
        num_subplots = len(pair_types)
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']  # Different colors for each subplot

        plt.figure(figsize=(15, 10))  # Adjust the figure size as needed
        for i, pair_type in enumerate(pair_types):
            plt.subplot(num_subplots, 1, i + 1)
            losses = self.pair_type_losses[pair_type] if pair_type != 'overall' else self.overall_losses
            plt.plot(epochs, losses, '-o', label=f'{pair_type} Loss', color=colors[i], markersize=3)
            plt.title(f'{pair_type} Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()  # Adjust subplots to fit into the figure area.

        file_name = f"combined_loss_plot_{self.save_tag}.png" if self.save_tag else "combined_loss_plot.png"
        file_path = f"./investigation/{file_name}"
        plt.savefig(file_path)
        plt.close()
        print(f"Saved combined loss plot at {file_path}")

    def save_percent_plot(self):
        # Plot the percentage of SEP-SEP pairs per epoch
        epochs = list(range(1, len(self.sep_sep_percentages) + 1))
        plt.figure()
        plt.plot(epochs, self.sep_sep_percentages, '-o', label='Percentage of SEP-SEP Pairs', markersize=3)
        plt.title(f'Percentage of SEP-SEP Pairs Per Epoch, Batch Size {self.batch_size}')
        plt.xlabel('Epoch')
        plt.ylabel('Percentage')
        plt.legend()
        plt.grid(True)
        # plt.show()  # or save the figure if preferred
        if self.save_tag:
            file_path = f"./investigation/percent_sep_sep_plot_{str(self.save_tag)}.png"
        else:
            file_path = f"./investigation/percent_sep_sep_plot.png"
        plt.savefig(file_path)
        plt.close()
        print(f"Saved plot at {file_path}")

    def save_sep_sep_loss_vs_frequency(self) -> None:
        """
        Plots the SEP-SEP loss against the SEP-SEP counts at the end of training.
        """
        plt.figure()
        plt.scatter(self.cumulative_sep_sep_counts, self.sep_sep_losses, c='blue', label='SEP-SEP Loss vs Frequency',
                    s=9)
        plt.title(f'SEP-SEP Loss vs Frequency, Batch Size {self.batch_size}')
        plt.xlabel('SEP-SEP Frequency')
        plt.ylabel('SEP-SEP Loss')
        plt.legend()
        plt.grid(True)

        if self.save_tag:
            file_path = f"./investigation/sep_sep_loss_vs_frequency_{self.save_tag}.png"
        else:
            file_path = "./investigation/sep_sep_loss_vs_frequency.png"

        plt.savefig(file_path)
        plt.close()
        print(f"Saved SEP-SEP Loss vs Counts plot at {file_path}")

    def save_slope_of_loss_vs_frequency(self) -> None:
        """
        Plots the slope of the change in SEP-SEP loss with respect to the change in SEP-SEP frequency vs epochs.
        """
        # Calculate the differences (delta) between consecutive losses and counts
        delta_losses = np.diff(self.sep_sep_losses)
        delta_counts = np.diff(self.cumulative_sep_sep_counts)

        # To avoid division by zero, we will replace zeros with a small value (epsilon)
        epsilon = 1e-8
        delta_counts = np.where(delta_counts == 0, epsilon, delta_counts)

        # Calculate the slope (change in loss / change in frequency)
        slopes = delta_losses / delta_counts

        # Prepare the epochs for x-axis, which are one less than the number of losses due to diff operation
        epochs = range(1, len(self.sep_sep_losses))

        plt.figure()
        plt.plot(epochs, slopes, '-o', label='Slope of SEP-SEP Loss vs Frequency', markersize=3)
        plt.title(f'Slope of SEP-SEP Loss vs Frequency Change Per Epoch, Batch Size {self.batch_size}')
        plt.xlabel('Epoch')
        plt.ylabel('Slope')
        plt.legend()
        plt.grid(True)

        if self.save_tag:
            file_path = f"./investigation/slope_sep_sep_loss_vs_frequency_{self.save_tag}.png"
        else:
            file_path = "./investigation/slope_sep_sep_loss_vs_frequency.png"

        plt.savefig(file_path)
        plt.close()
        print(f"Saved Slope of Loss vs Counts plot at {file_path}")

    # def _save_plot(self):
    #     plt.figure()
    #     plt.plot(self.epochs_10s, self.losses, '-o', label='Training Loss', markersize=3)
    #     plt.title(f'Training Loss, Batch Size {self.batch_size}')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.grid(True)
    #     if self.save_tag:
    #         file_path = f"./investigation/training_loss_plot_{str(self.save_tag)}.png"
    #     else:
    #         file_path = f"./investigation/training_loss_plot.png"
    #     plt.savefig(file_path)
    #     plt.close()
    #     print(f"Saved plot at {file_path}")

    # def save_loss_plot(self):
    #     """
    #     Saves a plot of the SEP loss at each epoch.
    #     """
    #     plt.figure()
    #     plt.plot(range(1, len(self.sep_sep_losses) + 1), self.sep_sep_losses, '-o', label='SEP Loss', markersize=3)
    #     plt.title(f'SEP Loss vs Batches, Batch Size {self.batch_size}')
    #     plt.xlabel('batches')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.grid(True)
    #     if self.save_tag:
    #         file_path = f"./investigation/sep_loss_plot_{self.save_tag}.png"
    #     else:
    #         file_path = "./investigation/sep_loss_plot.png"
    #     plt.savefig(file_path)
    #     plt.close()
    #     print(f"Saved SEP loss plot at {file_path}")
