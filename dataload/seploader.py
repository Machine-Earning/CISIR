import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from typing import Tuple, Dict, Any, Optional


class SEPLoader:
    def __init__(self):
        pass

    def load(self, file_path: str, num_folds: Optional[int] = None, num_shuffles: int = 3, SEED: int = 42,
             output_dir: str = 'data'):
        """
        Load the data from a CSV file, split it, shuffle it, and save the shuffled sets to CSV files.

        :param file_path: Path to the CSV file to load.
        :param num_folds: The number of folds to split the data into. If None, the data is not split into folds.
        :param num_shuffles: Number of times to shuffle each set.
        :param SEED: Optional random seed for reproducibility.
        :param output_dir: Directory to save the output CSV files.
        :return: None
        """

        # Set random seed for reproducibility
        np.random.seed(SEED)

        # Get the split data
        data_splits = self.read_data(file_path, num_folds)

        if num_folds is None:
            # Shuffle and save the single set of splits
            shuffled_data = self.shuffle_sets(data_splits[0], data_splits[1],
                                              data_splits[2], data_splits[3],
                                              data_splits[4], data_splits[5],
                                              num_shuffles, SEED)
            self.save_shuffled_data_to_csv(*shuffled_data, dir_name=output_dir)
        else:
            # Iterate over each fold
            for fold, data in data_splits.items():
                # Create directory for the fold if it doesn't exist
                fold_dir = os.path.join(output_dir, fold)
                os.makedirs(fold_dir, exist_ok=True)
                # Shuffle each fold's datasets the specified number of times
                shuffled_data = self.shuffle_sets(data['train_x'], data['train_y'],
                                                  data['val_x'], data['val_y'],
                                                  data['test_x'], data['test_y'],
                                                  num_shuffles, SEED)

                # Save the shuffled data to CSV files inside the fold directory
                self.save_shuffled_data_to_csv(*shuffled_data, dir_name=fold_dir)

    def load_from_dir(self, dir_path: str):
        """
        Load the shuffled data sets from CSV files in a specified directory.

        :param dir_path: Directory path containing the shuffled CSV files.
        :return: data = (train_x, train_y, val_x, val_y, test_x, test_y)
        """
        train_file = os.path.join(dir_path, 'shuffled_train.csv')
        val_file = os.path.join(dir_path, 'shuffled_val.csv')
        test_file = os.path.join(dir_path, 'shuffled_test.csv')

        data = self.read_shuffled_data_from_csv(train_file, val_file, test_file)

        return data

    def load_fold_from_dir(self, dir_path: str, fold_id: int):
        """
        Load the shuffled data sets from CSV files in a specified directory.

        :param fold_id: the fold to load data from
        :param dir_path: Directory path containing the shuffled CSV files.
        :return: fold(train_x, train_y, val_x, val_y, test_x, test_y)
        """
        train_file = os.path.join(dir_path + f'/fold_{fold_id}', 'shuffled_train.csv')
        val_file = os.path.join(dir_path + f'/fold_{fold_id}', 'shuffled_val.csv')
        test_file = os.path.join(dir_path + f'/fold_{fold_id}', 'shuffled_test.csv')

        fold_data = self.read_shuffled_data_from_csv(train_file, val_file, test_file)

        return fold_data

    def combine(self, *data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine an arbitrary number of datasets using *args. Assumes that x arrays are at even indices
        and y arrays are at odd indices in the arguments list.

        Parameters:
        - data (np.ndarray): Arbitrarily long list of numpy arrays. x arrays should be at even indices and y arrays at odd indices.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: Combined features and labels.
        """
        combined_x = np.concatenate([data[i] for i in range(0, len(data), 2)], axis=0)
        combined_y = np.concatenate([data[i] for i in range(1, len(data), 2)], axis=0)

        return combined_x, combined_y

    def read_data(self, file_path: str, num_folds=None):
        """
        Read data from a CSV file and split it into training, validation, and test sets.

        :param file_path: Path to the CSV file.
        :return: Shuffled versions of train_x, train_y, val_x, val_y, test_x, test_y
        """
        # Read the CSV file
        df = pd.read_csv(file_path)
        if num_folds is None:
            # Split the data into training, validation, and test sets
            return self.split_data(df)
        else:
            return self.split_data_folds(df, num_folds)

    def split_data(self, df):
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

    def split_data_folds(self, df: pd.DataFrame, num_folds: int = 3) -> Dict[str, Any]:
        """
        Splits the data into training, validation, and test sets according to the specified number of folds.
        Each fold's test set is picked sequentially rather than at random.

        :param df: DataFrame containing the data. Assumes 'log_peak_intensity' is the target column.
        :param num_folds: The number of folds to use for splitting the data.

        :return: A dictionary containing the split data for each fold.
        """
        # Sort the DataFrame by 'log_peak_intensity' in descending order
        df_sorted = df.sort_values(by='log_peak_intensity', ascending=False).reset_index(drop=True)

        # Initialize the dictionary to hold the splits for each fold
        fold_splits = {}

        # Perform the split for each fold
        for fold in range(num_folds):
            train_indices = []
            val_indices = []
            test_indices = []

            # Select test indices based on the current fold
            for i in range(fold, len(df_sorted), num_folds):
                test_indices.append(i)
                # The remaining indices of the group go to training
                train_indices.extend([i + offset for offset in range(1, num_folds) if i + offset < len(df_sorted)])

            # Group every 4 rows for validation set selection from the training set
            remaining_indices = list(set(range(len(df_sorted))) - set(test_indices))
            for i in range(0, len(remaining_indices), 4):
                group = remaining_indices[i: i + 4]
                if len(group) == 0:
                    continue
                val_idx = np.random.choice(group, 1)[0]
                val_indices.append(val_idx)
                remaining_indices = [idx for idx in remaining_indices if idx != val_idx]

            # Update the training indices after selecting validation indices
            train_indices = list(set(train_indices) - set(val_indices))

            # Extract the feature and target sets based on selected indices
            features = df_sorted.drop(columns=['log_peak_intensity']).to_numpy()
            target = df_sorted['log_peak_intensity'].to_numpy()

            fold_splits[f"fold_{fold + 1}"] = {
                'train_x': features[train_indices],
                'train_y': target[train_indices],
                'val_x': features[val_indices],
                'val_y': target[val_indices],
                'test_x': features[test_indices],
                'test_y': target[test_indices]
            }

        return fold_splits

    def shuffle_sets(self, train_x, train_y, val_x, val_y, test_x, test_y, num_shuffles=1, SEED=None):
        """
        Shuffle the data within each of the training, validation, and test sets.
        Shuffle each set `num_shuffles` times.

        :param:
        - train_x, train_y, val_x, val_y, test_x, test_y: Numpy arrays containing the features and labels
        - num_shuffles: Integer indicating the number of times to shuffle each set.
        - SEED: Optional random seed for reproducibility.

        :return:
        - Shuffled versions of train_x, train_y, val_x, val_y, test_x, test_y
        """
        for _ in range(num_shuffles):
            train_x, train_y = shuffle(train_x, train_y, random_state=SEED)
            val_x, val_y = shuffle(val_x, val_y, random_state=SEED)
            test_x, test_y = shuffle(test_x, test_y, random_state=SEED)

        return train_x, train_y, val_x, val_y, test_x, test_y

    def save_shuffled_data_to_csv(self, train_x, train_y, val_x, val_y, test_x, test_y, dir_name=''):
        """
        Save the shuffled data sets to CSV files within a specified directory.

        :param:
        - train_x, train_y, val_x, val_y, test_x, test_y: Numpy arrays containing the features and labels
        - dir_name: Optional directory name to add to the file paths.

        :return: None
        """
        # Create the directory if it doesn't exist
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # Combine features and labels and convert to DataFrame
        train_df = pd.DataFrame(np.column_stack([train_y, train_x]))
        val_df = pd.DataFrame(np.column_stack([val_y, val_x]))
        test_df = pd.DataFrame(np.column_stack([test_y, test_x]))

        # Full paths for saving the CSV files
        train_file = os.path.join(dir_name, 'shuffled_train.csv')
        val_file = os.path.join(dir_name, 'shuffled_val.csv')
        test_file = os.path.join(dir_name, 'shuffled_test.csv')

        # Save to CSV files
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)

    def read_shuffled_data_from_csv(self, train_file, val_file, test_file):
        """
        Read the shuffled data sets from CSV files.

        :param:
        - train_file, val_file, test_file: Paths to the CSV files for training, validation, and test sets.

        :return:
        - train_x, train_y, val_x, val_y, test_x, test_y: Numpy arrays containing the features and labels
        """
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)

        # Extract features and labels
        train_x = train_df.iloc[:, 1:].to_numpy()
        train_y = train_df.iloc[:, 0].to_numpy()
        val_x = val_df.iloc[:, 1:].to_numpy()
        val_y = val_df.iloc[:, 0].to_numpy()
        test_x = test_df.iloc[:, 1:].to_numpy()
        test_y = test_df.iloc[:, 0].to_numpy()

        return train_x, train_y, val_x, val_y, test_x, test_y
