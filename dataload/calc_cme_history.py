import pandas as pd
from typing import Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


def parse_datetime(date_str: str, format_str: str = "%m/%d/%Y %H:%M") -> datetime:
    """
    Parse a date string into a datetime object.

    Parameters:
    - date_str (str): The date string to parse.
    - format_str (str): The format of the date string.

    Returns:
    - datetime: The parsed datetime object.
    """
    return datetime.strptime(date_str, format_str)


def cme_counts_in_timeframe(df: pd.DataFrame, current_time: str, hours: int) -> int:
    """
    Count the number of CMEs within a certain number of hours before the given time,
    excluding the CME at the exact current time.
    """
    current_dt = parse_datetime(current_time)
    time_threshold = current_dt - timedelta(hours=hours)
    return df[df['CME_DONKI_time'].apply(
        lambda x: current_dt > parse_datetime(x) >= time_threshold)].shape[0]


def cme_counts_with_speed_threshold(df: pd.DataFrame, current_time: str, hours: int, speed: int) -> int:
    """
    Count the number of CMEs with speed over a certain threshold within a certain number of hours
    before the given time, excluding the CME at the exact current time.
    """
    current_dt = parse_datetime(current_time)
    time_threshold = current_dt - timedelta(hours=hours)
    return df[(df['CME_DONKI_time'].apply(
        lambda x: current_dt > parse_datetime(x) >= time_threshold)) & (
                      df['CME_DONKI_speed'] > speed)].shape[0]


def max_cme_speed_in_timeframe(df: pd.DataFrame, current_time: str, hours: int) -> int:
    """
    Get the maximum CME speed within a certain number of hours before the given time,
    excluding the CME at the exact current time.
    """
    current_dt = parse_datetime(current_time)
    time_threshold = current_dt - timedelta(hours=hours)
    subset_df = df[
        df['CME_DONKI_time'].apply(lambda x: current_dt > parse_datetime(x) >= time_threshold)]
    return subset_df['CME_DONKI_speed'].max() if not subset_df.empty else 0


def cme_statistics_for_row(df: pd.DataFrame, current_time: str) -> Tuple[int, int, int, int]:
    """
    Calculate various CME statistics for a specific row based on the current time.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing CME information.
    - current_time (str): The current time to use as a reference.

    Returns:
    - Tuple[int, int, int, int]: A tuple containing the number of CMEs in the past month, the number of CMEs in the past 9 hours,
                                 the number of CMEs with speed over 1,000 km/s in the past 9 hours, and the maximum CME speed in the past day.
    """
    past_month_count = cme_counts_in_timeframe(df, current_time, hours=30 * 24)
    past_9hr_count = cme_counts_in_timeframe(df, current_time, hours=9)
    past_9hr_speed_count = cme_counts_with_speed_threshold(df, current_time, hours=9, speed=1000)
    past_day_max_speed = max_cme_speed_in_timeframe(df, current_time, hours=24)

    return past_month_count, past_9hr_count, past_9hr_speed_count, past_day_max_speed


def calculate_average_duration(df: pd.DataFrame,
                               intensity_column: str = "Intensity",
                               index_column: str = 'Index',
                               datetime_column: str = "datetime",
                               threshold: float = 10.0,
                               start_year: int = None,
                               end_year: int = None,
                               debug: bool = False) -> pd.Timedelta:
    """
    Calculates the average duration between the start time (event index 1) and the peak time (event index 3)
    for events that meet a specified intensity threshold, within a given year range.

    Args:
        df (pd.DataFrame): DataFrame containing the event data.
        intensity_column (str): The name of the column in the DataFrame that contains intensity values.
        index_column (str): The name of the column in the DataFrame that contains index values for each event.
        datetime_column (str): The name of the column in the DataFrame that contains datetime values.
        threshold (float): The threshold value for the intensity above which events are considered.
        start_year (int, optional): The start year for filtering events. Defaults to None.
        end_year (int, optional): The end year for filtering events. Defaults to None.
        debug (bool, optional): whether to activate debugging logs

    Returns:
        pd.Timedelta: The average duration between the start and peak of the events
                      that meet the intensity threshold, within the specified year range.
    """

    # Filter the DataFrame for the specified year range if provided
    if start_year is not None or end_year is not None:
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        if start_year is not None:
            df = df[df[datetime_column].dt.year >= start_year]
        if end_year is not None:
            df = df[df[datetime_column].dt.year <= end_year]

    # Calculate the duration for each event
    durations = []
    for name, group in df.groupby(df.index // 4):
        if debug:
            # print name and group
            print(f'name: {name}, group: {group}')
        if group[group[index_column] == 3][intensity_column].iloc[0] > threshold:
            start_time = group[group[index_column] == 1][datetime_column].iloc[0]
            peak_time = group[group[index_column] == 3][datetime_column].iloc[0]
            if debug:
                print(f'start time: {start_time}, peak_time: {peak_time}')
            duration = pd.to_datetime(peak_time) - pd.to_datetime(start_time)
            durations.append(duration)

    # Calculate the average duration
    avg_duration = pd.to_timedelta(durations).mean()

    # Plot the distribution of durations if debug is True
    if debug:
        durations_hours = [d.total_seconds() / 3600 for d in durations]  # Convert durations to hours
        plt.figure(figsize=(10, 6))
        sns.histplot(durations_hours, kde=True, color="skyblue")
        plt.axvline(avg_duration.total_seconds() / 3600, color='red', linestyle='dashed', linewidth=1)
        plt.xlabel('Duration (hours)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Event Durations with Mean Duration')
        plt.grid(True)
        plt.show()

    return avg_duration