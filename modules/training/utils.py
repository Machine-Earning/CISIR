import glob
import os
import random
from typing import Dict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def trim_background(directory_path: str, threshold: float = 0.1) -> List[str]:
    """
    Processes all CSV files in a given directory, trimming the beginning of each file based on a 'Proton Intensity' condition.
    Trimming stops when the first row meeting a specified 'Proton Intensity' threshold is found. The trimmed files are saved
    with a '_trim' suffix in their name.

    Parameters:
    - directory_path: str - The path to the directory containing CSV files to be processed.
    - threshold: float - The proton intensity threshold used to determine the point of trimming.

    Returns:
    - List[str]: A list of filenames for the trimmed files.
    """
    # Find all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

    trimmed_files = []  # To store the paths of trimmed files

    for file_path in csv_files:
        data = pd.read_csv(file_path)

        # Convert 'Target Timestamp' to datetime objects
        data['Target Timestamp'] = pd.to_datetime(data['Target Timestamp'])

        # Initialize an empty list to store the index of the row where trimming should stop
        stop_index = None

        # Iterate through each row
        for i, row in data.iterrows():
            future_timestamp = row['Target Timestamp'] + pd.Timedelta(hours=3)
            future_intensity = data.loc[data['Target Timestamp'] == future_timestamp, 'Proton Intensity']

            if not future_intensity.empty and future_intensity.iloc[0] > threshold:
                stop_index = i  # Found the row meeting the condition, mark it for stopping the trim
                break  # Exit the loop as soon as the condition is met

        # If a stopping condition is found, trim the DataFrame up to that point; else, use the original data
        filtered_data = data.iloc[stop_index:] if stop_index is not None else data

        # Construct the new filename with '_trim' suffix
        trimmed_file_path = os.path.splitext(file_path)[0] + '_trim.csv'
        filtered_data.to_csv(trimmed_file_path, index=False)
        trimmed_files.append(trimmed_file_path)

    return trimmed_files


def plot_sorted_distributions(y_train, y_val, y_test, title='Sorted Distributions'):
    """
    Plots the sorted target variable distributions for the training, validation, and test sets.

    :param:
    - y_train: Numpy array containing the training set target variable
    - y_val: Numpy array containing the validation set target variable
    - y_test: Numpy array containing the test set target variable
    - title: Title for the plot
    """
    plt.figure(figsize=(12, 6))

    # Sort and plot the target variable for each set
    plt.plot(np.sort(y_train), label='Train', marker='o')
    plt.plot(np.sort(y_val), label='Validation', marker='x')
    plt.plot(np.sort(y_test), label='Test', marker='s')

    plt.xlabel('Index')
    plt.ylabel('log_peak_intensity')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def find_rand_bs(y, threshold=np.log(10), num_trials=1000, early_exit_ratio=0.95):
    """
    Optimized function to find the minimum batch size such that, when sampled at random,
    each batch is likely to contain at least 2 rows with target values greater than a given threshold.

    Parameters:
    - y: Numpy array containing the target values
    - threshold: The target threshold, default is ln(10)
    - num_trials: Number of trials to perform for each batch size
    - early_exit_ratio: The success ratio to achieve before stopping the trials for a given batch size

    Returns:
    - Minimum batch size that fulfills the condition
    """
    # Count the number of elements greater than the threshold
    count_above_threshold = np.sum(y > threshold)

    # If there are fewer than 2 elements above the threshold, return a message
    if count_above_threshold < 2:
        raise ValueError("There are not enough samples with target values greater than the threshold.")

    # Loop through possible batch sizes, starting from 2
    for batch_size in range(2, len(y) + 1):
        print(batch_size)
        success_count = 0

        for trial in range(num_trials):
            # Sample a random batch
            random_batch = random.sample(list(y), batch_size)

            # Check if the batch contains at least 2 elements greater than the threshold
            if np.sum(np.array(random_batch) > threshold) >= 2:
                success_count += 1

            # Early exit if success ratio is achieved
            if trial >= num_trials * 0.9 and success_count / (trial + 1) >= early_exit_ratio:
                return batch_size


def find_rand_bs_2(y, left_threshold, right_threshold, num_trials=1000, early_exit_ratio=0.95):
    """
    Function to find the minimum batch size such that, when sampled at random,
    each batch is likely to contain at least one row with a target value less than the left threshold
    and one row with a target value greater than the right threshold.

    Parameters:
    - y: Numpy array containing the target values
    - left_threshold: The left target threshold for filtering values less than this threshold
    - right_threshold: The right target threshold for filtering values greater than this threshold
    - num_trials: Number of trials to perform for each batch size
    - early_exit_ratio: The success ratio to achieve before stopping the trials for a given batch size

    Returns:
    - Minimum batch size that fulfills the condition
    """
    # Count the number of elements less than the left threshold and greater than the right threshold
    count_below_left_threshold = np.sum(y < left_threshold)
    count_above_right_threshold = np.sum(y > right_threshold)

    print(f"count_below_left_threshold: {count_below_left_threshold}")
    print(f"count_above_right_threshold: {count_above_right_threshold}")
    print("Sample values in y:", np.random.choice(y, 10, replace=False))  # Ensure sampling without replacement
    print("Min value in y:", np.min(y))
    print("Max value in y:", np.max(y))

    # If there are not enough elements to meet conditions, return a message
    if count_below_left_threshold < 1 or count_above_right_threshold < 1:
        raise ValueError("There are not enough samples meeting one or both threshold criteria.")

    # Loop through possible batch sizes, starting from 2 (minimum batch size to meet both conditions)
    for batch_size in range(4300, len(y) + 1):
        print(f'batch_size: {batch_size}')
        success_count = 0

        for trial in range(num_trials):
            # Sample a random batch
            random_batch = random.sample(list(y), batch_size)

            # Convert to Numpy array for easier computation
            batch_array = np.array(random_batch)

            # Check if the batch contains at least one element less than the left threshold
            # and one element greater than the right threshold
            if np.sum(batch_array < left_threshold) >= 1 and np.sum(batch_array > right_threshold) >= 1:
                success_count += 1

            # Early exit if success ratio is achieved
            if trial >= num_trials * 0.9 and success_count / (trial + 1) >= early_exit_ratio:
                return batch_size

    # If no batch size fulfills the condition within the given number of trials, raise an exception
    raise ValueError("Failed to determine a suitable batch size within the specified trials.")


def get_weight_path(weight_paths: Dict, slope: bool, cme: int = None):
    """
    Retrieves the weight path based on the given slope and cme conditions.

    :param weight_paths:  A dictionary containing the weight paths for different slope and cme conditions.
    :param slope: A boolean indicating whether slope is True or False.
    :param cme: An integer that can be -1, 0, or 500, indicating the cme value.
    :return: The corresponding weight path as a string, or None if not found.
    """
    if cme is None:
        return weight_paths.get(slope)
    else:
        return weight_paths.get((slope, cme))


if __name__ == '__main__':
    # Generate random target values
    # training_set_path = 'D:/College/Fall2023/sep-forecasting-research/data/electron_cme_data_split/training'

    # from modules.training.ts_modeling import build_dataset
    #
    # X_train, y_train = build_dataset(
    #     training_set_path,
    #     inputs_to_use=['e0.5', 'p'],
    #     add_slope=False,
    #     target_change=True)
    #
    # # Find the minimum batch size
    # # find the minimum batch size for the training set that satisfies the condition, using the optimized function
    # # min_batch_size = find_rand_bs(y_train, threshold=4.9, num_trials=2000,
    # #                                                   early_exit_ratio=0.99)
    #
    # min_batch_size = find_rand_bs_2(
    #     y_train,
    #     left_threshold=-1,
    #     right_threshold=1,
    #     num_trials=10000,
    #     early_exit_ratio=0.99)
    #
    # print(f"Minimum batch size: {min_batch_size}")

    root_path = "D:/College/Spring2024/SEP Forecasting Research/Dataset/electron_cme_v4/electron_cme_data_split_trim/"
    subpaths = ['training', 'subtraining', 'validation', 'testing']
    for subpath in subpaths:
        path = root_path + subpath
        trimmed_files = trim_background(path)
        print(trimmed_files)
