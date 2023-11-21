import pandas as pd
from typing import Tuple
from datetime import datetime, timedelta


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
