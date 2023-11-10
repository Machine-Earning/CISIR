import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os


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
                save_folder: str) -> None:
    """
    Plot the electron and proton flux data for a specific time range around a CME event in subplots and save the plots.

    :param df_flux: The flux DataFrame with both electron and proton fluxes.
    :param cme_time: The datetime of the CME event.
    :param start_hours_before: How many hours before the CME to start plotting.
    :param end_hours_after: How many hours after the CME to end plotting.
    :param save_folder: The folder path where the plots will be saved.
    """

    start_time = cme_time - timedelta(hours=start_hours_before)
    end_time = cme_time + timedelta(hours=end_hours_after)
    df_selected = df_flux.loc[start_time:end_time]

    # Create a 4x3 grid of subplots
    fig, axs = plt.subplots(4, 3, figsize=(20, 15))
    axs = axs.flatten()  # Flatten the 4x3 array to easily iterate over it

    # Plot electron fluxes in the first 4 subplots
    electron_flux_columns = [col for col in df_selected.columns if 'Electron_Flux' in col]
    for i, column in enumerate(electron_flux_columns):
        axs[i].plot(df_selected.index, df_selected[column], label=column)
        axs[i].legend()
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Flux (1/(cm^2 s sr MeV))')
        axs[i].set_title(f'{column}')
        axs[i].grid(True)

    # Plot proton fluxes in the next 7 subplots
    proton_flux_columns = [col for col in df_selected.columns if 'Proton_Flux' in col]
    for i, column in enumerate(proton_flux_columns, start=len(electron_flux_columns)):
        axs[i].plot(df_selected.index, df_selected[column], label=column, color='orange')
        axs[i].legend()
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Flux (1/(cm^2 s sr MeV))')
        axs[i].set_title(f'{column}')
        axs[i].grid(True)

    # Add a red vertical line for the CME time in each subplot before hiding the last one
    for ax in axs[:-1]:  # Exclude the last ax that will be hidden
        ax.axvline(x=cme_time, color='r', linestyle='--', label='CME Time')

    # Hide the last subplot (empty one) after plotting the red vertical lines
    axs[-1].axis('off')
    axs[-1].axvline(x=cme_time, color='r', linestyle='--', label='CME Time',
                    visible=False)  # Explicitly hide the line in the last subplot

    # Save the figure with a formatted filename based on the CME time and hours before/after
    filename = f"{cme_time.strftime('%Y%m%d_%H%M')}_{start_hours_before}hbefore_{end_hours_after}hafter.png"
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path)

    plt.tight_layout()
    plt.show()


# Define main function to execute the workflow
def main():
    # Load the data
    flux_data_path = 'D:/College/Fall2023/new_data/ephin5m.dat'
    # cme_time_str = '4/18/2014 13:09'
    cme_time_str = '8/14/2010 10:12'
    # Convert CME_DONKI_time to datetime
    cme_time = datetime.strptime(cme_time_str, '%m/%d/%Y %H:%M')

    df_flux = load_flux_data(flux_data_path)

    # Define save folder path
    save_folder = 'D:/College/Fall2023/New folder'

    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Call plot_fluxes with the save folder path
    plot_fluxes(df_flux, cme_time, start_hours_before=6, end_hours_after=6, save_folder=save_folder)


# Call the main function
main()
