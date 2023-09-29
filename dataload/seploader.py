import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


class SEPLoader:
    def __init__(self):
        pass

    def load(self, file_path: str, num_shuffles: int = 3, SEED: int = 42, outupt_dir: str = 'data'):
        """
        Load the data from a CSV file, split it, shuffle it, and save the shuffled sets to CSV files.

        :param file_path: Path to the CSV file to load.
        :param num_shuffles: Number of times to shuffle each set.
        :param SEED: Optional random seed for reproducibility.
        :param save_prefix: Optional prefix for the saved CSV filenames.
        :return: None
        """
        # Read and split the data
        train_x, train_y, val_x, val_y, test_x, test_y = self.read_data(file_path)

        # Shuffle the data
        shuffled_train_x, shuffled_train_y, shuffled_val_x, shuffled_val_y, shuffled_test_x, shuffled_test_y = \
            self.shuffle_sets(train_x, train_y, val_x, val_y, test_x, test_y, num_shuffles, SEED)

        # Save the shuffled data to CSV files
        self.save_shuffled_data_to_csv(shuffled_train_x, shuffled_train_y, shuffled_val_x, shuffled_val_y,
                                       shuffled_test_x, shuffled_test_y, dir_name=outupt_dir)

    def load_from_dir(self, dir_path: str):
        """
        Load the shuffled data sets from CSV files in a specified directory.

        :param dir_path: Directory path containing the shuffled CSV files.
        :return: train_x, train_y, val_x, val_y, test_x, test_y
        """
        train_file = os.path.join(dir_path, 'shuffled_train.csv')
        val_file = os.path.join(dir_path, 'shuffled_val.csv')
        test_file = os.path.join(dir_path, 'shuffled_test.csv')

        train_x, train_y, val_x, val_y, test_x, test_y = self.read_shuffled_data_from_csv(train_file, val_file,
                                                                                          test_file)

        return train_x, train_y, val_x, val_y, test_x, test_y

    def combine(self, train_x, train_y, val_x, val_y):
        """
        Combine the training and validation sets.

        :param train_x: Training features.
        :param train_y: Training labels.
        :param val_x: Validation features.
        :param val_y: Validation labels.
        :return: Combined features and labels.
        """
        combined_x = np.concatenate([train_x, val_x], axis=0)
        combined_y = np.concatenate([train_y, val_y], axis=0)

        return combined_x, combined_y

    def read_data(self, file_path: str):
        """
        Read data from a CSV file and split it into training, validation, and test sets.

        :param file_path: Path to the CSV file.
        :return: Shuffled versions of train_x, train_y, val_x, val_y, test_x, test_y
        """
        # Read the CSV file
        df = pd.read_csv(file_path)
        # Split the data into training, validation, and test sets
        return self.split_data(df)

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
