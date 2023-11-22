import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import matplotlib.dates as mdates
from typing import List


# Define a function to load and preprocess the electron and proton fluxes data
def load_flux_data(filepath: str) -> pd.DataFrame:
    """
    Load the electron and proton fluxes data from a file using spaces as delimiters.

    :param filepath: The path to the data file.
    :return: A pandas DataFrame with the preprocessed data.
    """
    # Read the file using whitespace as a delimiter, skipping the first four lines
    df = pd.read_csv(filepath, delim_whitespace=True, header=None, skiprows=4)

    # Define column names
    col_names = ['Year', 'Seconds_of_year', 'Electron_Flux_0.5MeV', 'Electron_Flux_1.8MeV',
                 'Electron_Flux_4.4MeV', 'Electron_Flux_7.4MeV', 'Proton_Flux_0.54MeV',
                 'Proton_Flux_1.37MeV', 'Proton_Flux_4.01MeV', 'Proton_Flux_6.10MeV',
                 'Proton_Flux_16.40MeV', 'Proton_Flux_33.00MeV', 'Proton_Flux_47.00MeV']

    # Assign the column names to the DataFrame
    df.columns = col_names

    # Convert the 'Year____Secs-of-year' to a datetime format
    df['Time'] = df.apply(lambda row: datetime(int(row['Year']), 1, 1) +
                                      timedelta(seconds=int(row['Seconds_of_year']))
    if pd.notna(row['Seconds_of_year']) else pd.NaT, axis=1)
    df.set_index('Time', inplace=True)

    # Replace invalid flux values with NaN
    df.replace(-9.9999998E+30, pd.NA, inplace=True)

    df = df.replace(pd.NA, np.nan)

    return df


# Define a function to plot the electron flux data
def plot_electron_flux(df_flux: pd.DataFrame, cme_time: datetime, start_hours_before: int = 2,
                       end_hours_after: int = 3) -> None:
    """
    Plot the electron flux data for a specific time range around a CME event.

    :param df_flux: The electron flux DataFrame.
    :param cme_time: The datetime of the CME event.
    :param start_hours_before: How many hours before the CME to start plotting.
    :param end_hours_after: How many hours after the CME to end plotting.
    """
    start_time = cme_time - timedelta(hours=start_hours_before)
    end_time = cme_time + timedelta(hours=end_hours_after)
    df_selected = df_flux.loc[start_time:end_time]

    plt.figure(figsize=(12, 6))
    for column in df_selected.columns:
        if 'Electron_Flux' in column:
            plt.plot(df_selected.index, df_selected[column], label=column)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Electron Flux (1/(cm^2 s sr MeV))')
    plt.title(f'Electron Flux from {start_time} to {cme_time}')
    plt.axvline(x=cme_time, color='r', linestyle='--', label='CME Time')
    plt.grid(True)
    plt.show()


def plot_fluxes(df_flux: pd.DataFrame, cme_time: datetime, start_hours_before: int, end_hours_after: int,
                save_folder: str, title_suffix: str = '') -> None:
    """
    Plot the electron and proton flux data for a specific time range around a CME event in subplots and save the plots.

    :param df_flux: The flux DataFrame with both electron and proton fluxes.
    :param cme_time: The datetime of the CME event.
    :param start_hours_before: How many hours before the CME to start plotting.
    :param end_hours_after: How many hours after the CME to end plotting.
    :param save_folder: The folder path where the plots will be saved.
    :param title_suffix: Additional title information to be appended.
    """

    start_time = cme_time - timedelta(hours=start_hours_before)
    end_time = cme_time + timedelta(hours=end_hours_after)
    df_selected = df_flux.loc[start_time:end_time]

    # Create a 4x3 grid of subplots
    fig, axs = plt.subplots(4, 3, figsize=(24, 18))
    axs = axs.flatten()  # Flatten the 4x3 array to easily iterate over it

    # Define the electron and proton flux columns
    electron_flux_columns = [col for col in df_selected.columns if 'Electron_Flux' in col]
    proton_flux_columns = [col for col in df_selected.columns if 'Proton_Flux' in col]

    # Format the CME date and time into a string
    cme_date_str = cme_time.strftime('%m/%d/%Y')
    cme_time_str = cme_time.strftime('%H:%M')

    # Define the date format for the x-axis labels
    date_format = mdates.DateFormatter('%H:%M')

    # Iterate over subplots to set titles, date format, and handle no data cases
    for i, ax in enumerate(axs):
        ax.xaxis.set_major_formatter(date_format)
        ax.grid(True)  # Enable grid
        ax.set_xlabel('Time')
        ax.set_ylabel('Flux (1/(cm^2 s sr MeV))')

        # Determine whether to plot data or display 'No Data'
        if i < len(electron_flux_columns):
            column = electron_flux_columns[i]
            ax.set_title(f'Electron Flux: {column} on {cme_date_str}{title_suffix}', fontsize=10)
            plot_color = 'blue'
        elif i < len(proton_flux_columns) + len(electron_flux_columns):
            column = proton_flux_columns[i - len(electron_flux_columns)]
            ax.set_title(f'Proton Flux: {column} on {cme_date_str}{title_suffix}', fontsize=10)
            plot_color = 'orange'  # Different color for proton flux
        else:
            ax.axis('off')
            continue  # Skip the rest of the loop for this ax

        # Check for valid data
        if df_selected[column].notna().any():
            ax.plot(df_selected.index, df_selected[column], label=column, color=plot_color)
            ax.axvline(x=cme_time, color='r', linestyle='--', label=f'CME Time: {cme_time_str}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14, color='gray')

    # Use tight_layout with padding, or subplots_adjust for more control
    plt.tight_layout(pad=1.8)  # Adjust padding as needed
    # For more control, you can use:
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    # Save the figure with adjusted layout
    # Save the figure with a formatted filename based on the CME time and hours before/after
    filename = f"{cme_time.strftime('%m_%d_%Y_%H_%M')}_{start_hours_before}hbefore_{end_hours_after}hafter{title_suffix.replace(' ', '_').replace(':', '_')}.png"
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path)

    # Close the figure after saving
    plt.close(fig)


def plot_high_intensity_events(data_path: str, threshold: float, flux_data: pd.DataFrame, save_folder: str):
    # Read the CSV file
    sep_data = pd.read_csv(data_path, usecols=['peak_intensity', 'CME_DONKI_time'])

    # Filter events based on the threshold
    high_intensity_events = sep_data[sep_data['peak_intensity'] > threshold]

    # Iterate over each event and plot
    for index, event in high_intensity_events.iterrows():
        # Parse the CME time
        cme_time = datetime.strptime(event['CME_DONKI_time'], '%m/%d/%Y %H:%M')

        # Update filename and title with peak intensity
        peak_intensity_str = str(event['peak_intensity'])
        filename_suffix = f"_intensity_{peak_intensity_str.replace('.', '_')}"
        plot_title_suffix = f" - Peak Intensity: {peak_intensity_str}"

        # Call the plot_fluxes function with modified title and filename
        # modified_save_folder = os.path.join(save_folder, filename_suffix)
        # os.makedirs(modified_save_folder, exist_ok=True)
        plot_fluxes(flux_data, cme_time, 6, 6, save_folder, plot_title_suffix)


# def add_electron_flux_features_to_SEP(df_flux: pd.DataFrame, sep_df: pd.DataFrame, channels: list,
#                                       output_filepath: str):
#     # Iterate over each row in the SEP DataFrame
#     for index, row in sep_df.iterrows():
#         cme_time_str = row['CME_DONKI_time']
#         cme_time = datetime.strptime(cme_time_str, '%m/%d/%Y %H:%M')
#         start_time = cme_time - timedelta(hours=2)
#
#         # Extract the relevant time range for this event
#         df_selected = df_flux.loc[start_time:cme_time]
#
#         # Loop through each channel and add the columns to sep_df
#         for channel in channels:
#             # Resample the data to 5-minute intervals
#             resampled_data = df_selected[channel].resample('5T').mean()
#
#             # Create column names based on time offset from CME and add them to sep_df
#             for i, value in enumerate(resampled_data):
#                 time_offset = (cme_time - resampled_data.index[i]).total_seconds() / 60  # minutes
#                 col_name = f'{channel}_{int(time_offset)}min'
#                 sep_df.at[index, col_name] = value
#
#     # Save the updated DataFrame to a file
#     sep_df.to_csv(output_filepath, index=False)

# def add_electron_flux_features_to_SEP(df_flux: pd.DataFrame, sep_df: pd.DataFrame, channels: list, output_filepath: str):
#     # Initialize a list to hold temporary DataFrames
#     temp_dfs = []
#
#     # Iterate over each row in the SEP DataFrame
#     for index, row in sep_df.iterrows():
#         cme_time_str = row['CME_DONKI_time']
#         cme_time = datetime.strptime(cme_time_str, '%m/%d/%Y %H:%M')
#         start_time = cme_time - timedelta(hours=2)
#
#         # Extract the relevant time range for this event
#         df_selected = df_flux.loc[start_time:cme_time]
#
#         # Create a dictionary to hold features for this event
#         temp_features = {}
#
#         # Loop through each channel and add the features to the dictionary
#         for channel in channels:
#             # Resample the data to 5-minute intervals
#             resampled_data = df_selected[channel].resample('5T').mean()
#
#             # Create column names and add them to the dictionary
#             for i, value in enumerate(resampled_data):
#                 time_offset = (cme_time - resampled_data.index[i]).total_seconds() / 60  # minutes
#                 col_name = f'{channel}_{int(time_offset)}min'
#                 temp_features[col_name] = value
#
#         # Convert the dictionary to a DataFrame and append it to the list
#         temp_df = pd.DataFrame(temp_features, index=[index])
#         temp_dfs.append(temp_df)
#
#     # Concatenate all temporary DataFrames with the original sep_df
#     features_df = pd.concat(temp_dfs, axis=0)
#     updated_df = pd.concat([sep_df, features_df], axis=1)
#
#     # Save the updated DataFrame to a file
#     updated_df.to_csv(output_filepath, index=False)

def add_electron_flux_features_to_SEP(df_flux: pd.DataFrame, sep_df: pd.DataFrame,
                                      channels: List[str], output_filepath: str) -> None:
    """
    Adds electron flux features to the SEP events DataFrame.

    For each CME event in the SEP DataFrame, this function extracts electron flux data
    for the specified channels for two hours leading up to the CME. The data is sampled
    in 5-minute intervals. This information is then added as new columns to the SEP DataFrame.

    Args:
        df_flux (pd.DataFrame): DataFrame containing electron and proton flux data.
        sep_df (pd.DataFrame): DataFrame containing SEP events data.
        channels (List[str]): List of electron flux channels to include (e.g., 'Electron_Flux_0.5MeV').
        output_filepath (str): File path where the updated DataFrame will be saved.

    Returns:
        None: The function saves the updated DataFrame to the specified filepath.
    """

    # Initialize a list to hold temporary DataFrames
    temp_dfs = []

    # Iterate over each row in the SEP DataFrame
    for index, row in sep_df.iterrows():
        cme_time_str = row['CME_DONKI_time']
        cme_time = datetime.strptime(cme_time_str, '%m/%d/%Y %H:%M')
        start_time = cme_time - timedelta(hours=2)

        # Extract the relevant time range for this event
        df_selected = df_flux.loc[start_time:cme_time]

        # Create a dictionary to hold features for this event
        temp_features = {}

        # Loop through each channel and add the features to the dictionary
        for channel in channels:
            # Resample the data to 5-minute intervals
            resampled_data = df_selected[channel].resample('5T').mean()

            # Create column names and add them to the dictionary
            for i, value in enumerate(resampled_data):
                col_name = f'{channel}_t-{24 - i}'  # "t-x" format
                temp_features[col_name] = value

        # Convert the dictionary to a DataFrame and append it to the list
        temp_df = pd.DataFrame(temp_features, index=[index])
        temp_dfs.append(temp_df)

    # Concatenate all temporary DataFrames with the original sep_df
    features_df = pd.concat(temp_dfs, axis=0)
    updated_df = pd.concat([sep_df, features_df], axis=1)

    # Save the updated DataFrame to a file
    updated_df.to_csv(output_filepath, index=False)


# Define main function to execute the workflow
def main():
    # Load the data
    # flux_data_path = 'D:/College/Fall2023/new_data/ephin5m.dat'
    # cme_time_str = '4/18/2014 13:09'  # 59.031 pfu	1	4/18/2014 13:09
    # cme_time_str = '8/14/2010 10:12'  # 14.608 pfu	1	8/14/2010 10:12 barely
    # cme_time_str = '6/21/2015 2:48'  # 961.13 pfu	6/21/2015 2:48
    # cme_time_str = '1/1/2016 23:12'  # 20.623	1	1/1/2016 23:12
    # cme_time_str = '11/1/2014 5:12'  # 10.584	1	11/1/2014 5:12
    # cme_time_str = '1/23/2012 4:00'  # 6198.6	1	1/23/2012 4:00
    # cme_time_str = '3/7/2012 0:36'  # 5919.2	1	3/7/2012 0:36

    data_path = 'D:/College/Fall2023/new_data/SEP10MeV_Features.csv'
    threshold = 10  # pfu so SEPs only
    flux_data_path = 'D:/College/Fall2023/new_data/ephin5m.dat'
    # save_folder = 'D:/College/Fall2023/High_Intensity_Events/'
    # # Ensure the save folder exists
    # os.makedirs(save_folder, exist_ok=True)
    # df_flux = load_flux_data(flux_data_path)
    # plot_high_intensity_events(data_path, threshold, df_flux, save_folder)

    df_flux = load_flux_data(flux_data_path)
    sep_df = pd.read_csv(data_path)
    channels = ['Electron_Flux_0.5MeV', 'Electron_Flux_1.8MeV', 'Electron_Flux_4.4MeV']
    output_filepath = 'D:/College/Fall2023/new_data/Updated_SEP10MeV_Features2.csv'

    add_electron_flux_features_to_SEP(df_flux, sep_df, channels, output_filepath)


# Call the main function
main()
