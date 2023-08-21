import argparse

def create_requirements(input_file):
    """
    Reads package information from a text file and creates a requirements.txt file.

    :param input_file: The path to the text file containing the package information.
    """
    with open(input_file, "r") as file:
        # Read the lines and skip the header
        lines = file.readlines()[2:]

    # Open the output file
    with open("requirements.txt", "w") as file:
        # Iterate through the lines and write the package name and version
        for line in lines:
            name, version = line.split()[:2]
            file.write(f"{name}=={version}\n")

    print("Requirements file created successfully.")

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Convert package information to requirements.txt")
    parser.add_argument("input_file", help="Path to the text file containing the package information")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the input file
    create_requirements(args.input_file)

if __name__ == "__main__":
    main()

