import csv
import math
import argparse


def readCSVFile(csvFile):
    """
    Read a CSV file and store it in a list
    
    Output: List[Dict[str: str]]
        The read CSV file. Row 1 of the CSV file matches with List[0]. Each row is a dictionary mapping fieldnames to their values as strings.
    """
    rows = []
    with open(csvFile, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)

    return rows


def writeCSVFile(data, csvFilename, fieldnames):
    """
    Write a CSV file to the provided csv filename.
    data : List[Dict[str: any]]
        The CSV data to write. Each index of the list should be a row of data. Each row of data should contain a mapping of fieldname to cell contents.
    csvFilename: str
        The path of the CSV file to write. If already existing, it will be overwritten
    fieldnames: List[str]
        The columns of the resulting CSV. Each row of the data must contain each of these as a key in their dictionary mapped to something otherwise an error will be thrown.
    """
    with open(csvFilename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for elem in data:
            row = {}
            for feature in fieldnames:
                row[feature] = elem[feature]
            writer.writerow(row)
    return


def calculate_connection_angle(data, newFeatureName):
    """
    Calculates the connection angle in degrees for each row from its CME features
    data : List[Dict[str: str]]
        The CSV file data where each row is a dictionary in the list mapping fieldnames to contents.
        Each row will be updated in place.
    newFeatureName : str
        The name of the connection angle feature to write. This is the fieldname heading that will be used for the calculated feature
    """
    angular_speed_of_sun = 360 / (27.27 * 86400)
    au = 1.5 * math.pow(10, 8)
    for elem in data:
        # connection angle = arccos(sin(theta_1) * sin(theta_2) + cos(theta_1) * cos(theta_2) * cos(phi_1 - phi_2))
        # theta_1 = latitude (in degrees)
        # phi_1 = longitude (in degrees)
        # theta_2 = 0
        # phi_2 = angular_speed_of_sun * 1 AU / solar_wind_speed (in degrees)
        # angular_speed_of_sun = 360 / 27.27 * 86400) degrees/second
        # 1 AU = 1.5 * 10^8 km
        theta_1 = float(elem["CME_DONKI_latitude"])
        phi_1 = float(elem["CME_DONKI_longitude"])
        theta_2 = 0
        phi_2 = angular_speed_of_sun * au / float(elem["solar_wind_speed"])

        # Optional column if you want to see phi 2 for each row
        # elem["phi_2_degrees"] = phi_2

        # math.acos wants RADIANS not DEGREES
        # math.sin wants RADIANS not DEGREES
        # math.cos wants RADIANS not DEGREES
        theta_1_rad = math.radians(theta_1)
        phi_1_rad = math.radians(phi_1)
        theta_2_rad = math.radians(theta_2)
        phi_2_rad = math.radians(phi_2)
        # math.acos only returns positive values. The connection angle sometimes should be negative
        if phi_1 >= phi_2 - 180 and phi_1 <= phi_2:
            sign = -1
        else:
            sign = 1

        # math.acos returns RADIANS but we want DEGREES
        connection_angle_rad = sign * math.acos(
            math.sin(theta_1_rad) * math.sin(theta_2_rad) + math.cos(theta_1_rad) * math.cos(theta_2_rad) * math.cos(
                phi_1_rad - phi_2_rad))

        elem[newFeatureName] = math.degrees(connection_angle_rad)


def calculate_diffusive_shock_v(mev=10):
    """
    Calculate the v component of the diffusive shock equation
    mev : int, optional
        The Mega electronvolts value, default is 10.
    Output : float
        The v component of the diffusive shock equation for 100 MeV
    """
    # v (Particle speed for 100 MeV protons): (3 * v_in_km/s) * sqrt(1 - (1/((100 MeV + 938 MeV) / 938 MeV))^2)
    # Particle speed for 100 MeV protons: 138425 km/s
    # Particle speed for 10 MeV protons: 43774 km/s
    if mev == 100:
        v = 138425
    elif mev == 10:
        v = 43774
    else:
        raise ValueError("Unsupported MeV value")
    v_c = 3 * v  # math.pow(10, 5)  # km/s
    v_gamma = (mev + 938.0) / 938.0
    return v_c * math.sqrt(1 - math.pow((1.0 / v_gamma), 2))


def calculate_diffusive_shock(data, newFeatureName, mev=10):
    """
    Calculate the diffusive shock for each row from its CME features
    data : List[Dict[str: str]]
        The CSV file data where each row is a dictionary in the list mapping fieldnames to contents.
        Each row will be updated in place.
    mev : int, optional
        The Mega electronvolts value, default is 10.
    newFeatureName : str
        The name of the diffusive shock feature to write. This is the fieldname heading that will be used for the calculated feature
    """
    # diffusive_shock = term_1 * term_2 * term_3 * term_4 * term_5
    # term_1 = N
    # term_2 = v
    # term_3 = 1 / (gamma - 1)
    # term_4 = 1 / (1 + term_4_sub_1)^(k+1)
    # term_4_sub_1 = Vinj^2 / (k * Vth^2)
    # term_5 = (Vinj / v)^(gamma + 1)

    # N (shock efficiency): 0.1
    # v (Particle speed for 100 MeV protons): (3 * 10^5 km/s) * sqrt(1 - (1/((100 MeV + 938 MeV) / 938 MeV))^2)
    # gamma: (if M > 1.1): (4 * M^2) / (M^2 - 1)
    # gamma: (else): (4 * 1.1^2) / (1.1^2 - 1)
    # M = Vsh / Va
    # Vsh (shock speed or Linear Speed from DONKI (not CDAW))
    # Va (Alven speed): 600 km/s
    # Vinj = 2.5 * Vsh
    # k (distribution parameter): 2
    # Vth (proton thermal speed): 150 km/s
    for elem in data:
        N = 0.1
        v = calculate_diffusive_shock_v(mev)
        vsh = float(elem["CME_DONKI_speed"])
        va = float(600)
        vinj = 2.5 * vsh
        k = float(2)
        vth = float(150)
        m = vsh / va
        if m > 1.1:
            gamma = (4 * math.pow(m, 2)) / (math.pow(m, 2) - 1)
        else:
            gamma = (4 * math.pow(1.1, 2)) / (math.pow(1.1, 2) - 1)

        term_1 = N
        term_2 = v
        term_3 = 1 / (gamma - 1)
        term_4_sub_1 = math.pow(vinj, 2) / (k * math.pow(vth, 2))
        term_4 = 1 / math.pow(1 + term_4_sub_1, k + 1)
        term_5 = math.pow(vinj / v, gamma + 1)
        diffusive_shock = term_1 * term_2 * term_3 * term_4 * term_5

        elem[newFeatureName] = diffusive_shock


# Complete the calculate_richardson function according to the given equation
def calculate_richardson(data, newFeatureName):
    """
    Calculate the Richardson for each row from its connection angle feature.

    data : List[Dict[str: str]]
        The CSV file data where each row is a dictionary in the list mapping fieldnames to contents.
        Each row will be updated in place.

    newFeatureName : str
        The name of the Richardson feature to write. This is the fieldname heading that will be used for the calculated feature.
    """
    # Richardson = - (phi ^ 2) / (2*(43^2))
    for elem in data:
        phi = float(elem["connection_angle_degrees"])
        richardson = - (math.pow(phi, 2)) / (2 * math.pow(43, 2))
        elem[newFeatureName] = richardson


def sortDataIndex(e):
    """
    Function that return the sorting order of the provided element based on its index value
    e : Dict[str: any]
        The element
    Output : int
        The sorting order for this element
    """
    return int(e["index"])


def main():
    # Parse the arguments from the user
    parser = argparse.ArgumentParser()
    parser.add_argument("data_filename", help="The CSV with the raw features.")
    parser.add_argument("-o", "--output", default="a.csv",
                        help="The output filename. The output will be a CSV file, so you should provide a file that ends in CSV ex: a.csv")
    args = parser.parse_args()

    # Read the CSV file with features
    data = readCSVFile(args.data_filename)

    # Extract out the feature names
    fieldnames = list(data[0].keys())

    # Calculate diffusive shock and update the data in-place. Add feature name to fieldnames list
    diffusive_shock_feature_name = "diffusive_shock"
    calculate_diffusive_shock(data, diffusive_shock_feature_name, mev=10)
    fieldnames.append(diffusive_shock_feature_name)

    # Calculate connection angle and update the data in-place. Add feature name to fieldnames list
    connection_angle_feature_name = "connection_angle_degrees"
    calculate_connection_angle(data, connection_angle_feature_name)
    fieldnames.append(connection_angle_feature_name)

    # Calculate Richardson and update the data in-place. Add feature name to fieldnames list
    richardson_feature_name = "richardson_value"
    calculate_richardson(data, richardson_feature_name)
    fieldnames.append(richardson_feature_name)

    # Resort to the provided order (assumes it was provided by index)
    data.sort(key=sortDataIndex)

    # Write the updated in-place data to the output file
    writeCSVFile(data, args.output, fieldnames)


if __name__ == "__main__":
    main()
