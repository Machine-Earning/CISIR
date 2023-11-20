import pandas as pd
import math
import argparse
import numpy as np


def readCSVFile(csvFile: str) -> pd.DataFrame:
    """
    Reads a CSV file and returns it as a DataFrame.

    Args:
        csvFile (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Data read from the CSV file.
    """
    return pd.read_csv(csvFile)


def writeCSVFile(df: pd.DataFrame, csvFilename: str) -> None:
    """
    Writes a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to write.
        csvFilename (str): Path where the CSV file will be written.

    Returns:
        None
    """
    df.to_csv(csvFilename, index=False)


def calculate_connection_angle(df: pd.DataFrame) -> None:
    """
    Calculates the connection angle in degrees and stores it in the DataFrame.

    connection angle = arccos(sin(theta_1) * sin(theta_2) + cos(theta_1) * cos(theta_2) * cos(phi_1 - phi_2))
    theta_1 = latitude (in degrees)
    phi_1 = longitude (in degrees)
    theta_2 = 0
    phi_2 = angular_speed_of_sun * 1 AU / solar_wind_speed (in degrees)
    angular_speed_of_sun = 360 / 27.27 * 86400) degrees/second
    1 AU = 1.5 * 10^8 km
    convert between radians and degrees due to arccos working in radians and
    we want degrees.
    Args:
        df (pd.DataFrame): DataFrame containing the required columns for calculation.

    Returns:
        None
    """
    angular_speed_of_sun = 360 / (27.27 * 86400)  # in degrees/second
    au = 1.5 * math.pow(10, 8)  # in km

    theta_1_rad = pd.Series(df['CME_DONKI_latitude']).apply(math.radians)
    phi_1_rad = pd.Series(df['CME_DONKI_longitude']).apply(math.radians)
    theta_2_rad = 0  # In radians
    phi_2 = angular_speed_of_sun * au / df['solar_wind_speed']  # in degrees
    phi_2_rad = pd.Series(phi_2).apply(math.radians)  # converting to radians

    # Determine the sign of the connection angle
    condition = (phi_1_rad >= phi_2_rad - math.pi) & (phi_1_rad <= phi_2_rad)
    sign = pd.Series(np.where(condition, -1, 1))

    # Calculate connection angle using the correct formula
    df['connection_angle_degrees'] = sign * pd.Series(
        np.arccos(
            np.sin(theta_1_rad) * np.sin(theta_2_rad) +
            np.cos(theta_1_rad) * np.cos(theta_2_rad) *
            np.cos(phi_1_rad - phi_2_rad)
        )
    ).apply(math.degrees)


def calculate_diffusive_shock(df: pd.DataFrame, mev: int = 10) -> None:
    """
    Calculates the diffusive shock and stores it in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the required columns for calculation.
        mev (int, optional): Mega electronvolts value, default is 10.

    Raises:
        ValueError: If an unsupported MeV value is provided.

    Returns:
        None
    """

    if mev not in [10, 100]:
        raise ValueError("Unsupported MeV value")

    # Calculate the v value based on MeV
    #  v (Particle speed for mev MeV protons): (3 * raw_v_in_km/s) * sqrt(1 - (1/((mev MeV + 938 MeV) / 938 MeV))^2)
    #     Particle speed for 100 MeV protons: 138425 km/s
    #     Particle speed for 10 MeV protons: 43774 km/s
    raw_v = 138425 if mev == 100 else 43774
    v_c = 3 * raw_v
    v_gamma = (mev + 938.0) / 938.0
    v = v_c * math.sqrt(1 - math.pow((1.0 / v_gamma), 2))

    # diffusive_shock = term_1 * term_2 * term_3 * term_4 * term_5
    # term_1 = N
    # term_2 = v
    # term_3 = 1 / (gamma - 1)
    # term_4 = 1 / (1 + term_4_sub_1)^(k+1)
    # term_4_sub_1 = Vinj^2 / (k * Vth^2)
    # term_5 = (Vinj / v)^(gamma + 1)

    # N (shock efficiency): 0.1
    # v
    # gamma: (if M > 1.1): (4 * M^2) / (M^2 - 1)
    # gamma: (else): (4 * 1.1^2) / (1.1^2 - 1)
    # M = Vsh / Va
    # Vsh (shock speed or Linear Speed from DONKI (not CDAW))
    # Va (Alven speed): 600 km/s
    # Vinj = 2.5 * Vsh
    # k (distribution parameter): 2
    # Vth (proton thermal speed): 150 km/s

    # Extract relevant columns from the DataFrame
    N = 0.1
    vsh = df["CME_DONKI_speed"]
    va = 600.0
    # Additional calculations
    vinj = 2.5 * vsh
    k = 2.0
    vth = 150.0
    m = vsh / va

    # Conditional gamma calculation
    m = m.clip(lower=1.1)  # clip lower values of m to 1.1
    gamma = (4 * m.pow(2)) / (m.pow(2) - 1)

    # terms
    term_1 = N
    term_2 = v
    term_3 = 1 / (gamma - 1)
    term_4_sub_1 = vinj.pow(2) / (k * math.pow(vth, 2))
    term_4 = 1 / (1 + term_4_sub_1).pow(k + 1)
    term_5 = (vinj / v).pow(gamma + 1)

    # Calculate diffusive shock
    df['diffusive_shock'] = term_1 * term_2 * term_3 * term_4 * term_5


def calculate_richardson(df: pd.DataFrame) -> None:
    """
    Calculates the Richardson (half) value and stores it in the DataFrame.
    Equation: R = -conn_angle^2 / (2*43^2) where conn_angle and 43 in degrees.
    Args:
        df (pd.DataFrame): DataFrame containing the required columns for calculation.

    Returns:
        None
    """
    df['half_richardson_value'] = -(df['connection_angle_degrees'] ** 2) / (2 * (43 ** 2))


def main() -> None:
    """
    Main program
    Parse arguments
    Calculates the necessary quantities and stores them in the DataFrame inplace
    outputs to a csv file
    """
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("data_filename", help="The CSV with the raw features.")
        parser.add_argument("-o", "--output", default="a.csv",
                            help="The output filename. The output will be a CSV file, "
                                 "so you should provide a file that ends in CSV ex: a.csv")
        args = parser.parse_args()

        df = readCSVFile(args.data_filename)

        calculate_diffusive_shock(df)
        calculate_connection_angle(df)
        calculate_richardson(df)

        df.sort_values(by='index', inplace=True)

        writeCSVFile(df, args.output)

    except ModuleNotFoundError:
        print("It seems some required Python packages are missing. Please install them by running:")
        print("pip install numpy pandas argparse")


if __name__ == "__main__":
    main()
