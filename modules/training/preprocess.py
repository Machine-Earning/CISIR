import os

import numpy as np
import pandas as pd


def update_csv_files_with_delta_log_intensity(directory_path: str):
    """
    Update all CSV files in the given directory with a new column called 'delta_log_Intensity'.
    :param directory_path:  The path to the directory containing the CSV files.
    :return:            None
    """
    # List all files in the given directory
    for filename in os.listdir(directory_path):
        # Check if the file is a CSV
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)

            print(f"Updating {filename}...")
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)

            print('Shape before: ')
            print(df.shape)

            # Check if the required columns exist in the DataFrame
            if 'Proton Intensity' in df.columns and 'p_t' in df.columns:
                # Check for NaN values in the columns
                if df['Proton Intensity'].isnull().any() or df['p_t'].isnull().any():
                    raise ValueError(f"The file {filename} contains NaN values.")

                # Calculate the natural log of the 'Proton Intensity' and 'p_t' columns using log1p
                df['delta_log_Intensity'] = np.log1p(df['Proton Intensity']) - np.log1p(df['p_t'])

                # Save the updated DataFrame back to the CSV file
                df.to_csv(file_path, index=False)

                print(f"Updated {filename}.")
                # print the number of rows and columns
                print('Shape after: ')
                print(df.shape)
            else:
                raise ValueError(f"The file {filename} does not contain the required columns.")
        else:
            print(f"Skipping {filename}, as it's not a CSV file.")

    print("All CSV files have been updated.")


if __name__ == '__main__':
    print("Updating CSV files with delta log intensity...")
    paths = ['D:/College/Fall2023/sep-forecasting-research/data/electron_cme_data_split/training',
             'D:/College/Fall2023/sep-forecasting-research/data/electron_cme_data_split/subtraining',
             'D:/College/Fall2023/sep-forecasting-research/data/electron_cme_data_split/validation',
             'D:/College/Fall2023/sep-forecasting-research/data/electron_cme_data_split/testing']
    try:
        for path in paths:
            update_csv_files_with_delta_log_intensity(path)
    except ValueError as e:
        print(e)
    print("Done.")
