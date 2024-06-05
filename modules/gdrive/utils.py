import os
from typing import Union, List

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def authenticate_gdrive(credentials_file: str = "mycreds.txt") -> GoogleDrive:
    """
    Authenticates and returns a Google Drive client.

    This function handles the OAuth2 authentication process for Google Drive. It loads the credentials
    from a file, refreshes them if they have expired, or performs a full authentication if no valid
    credentials are found. The credentials are saved to the specified file for future use.

    :param credentials_file: The path to the file where credentials are stored. Default is "mycreds.txt".
    :return: An authenticated Google Drive client.
    """
    # Authenticate and create the PyDrive client
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile(credentials_file)
    if not gauth.credentials:
        gauth.LocalWebserverAuth()  # Creates local webserver and handles authentication.
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile(credentials_file)

    return GoogleDrive(gauth)


def upload_to_drive(file_paths: Union[str, List[str]], drive_folder_id: str, delete_local: bool = False) -> List[str]:
    """
    Uploads a file or a list of files to a specified folder on Google Drive and optionally deletes the local file(s)
    after a successful upload.

    :param file_paths: The path(s) of the file(s) to upload. Can be a single file path (str) or a list of file paths (List[str]).
    :param drive_folder_id: The ID of the destination folder on Google Drive where the file(s) will be uploaded.
    :param delete_local: A boolean flag indicating whether to delete the local file(s) after a successful upload. Default is False.
    :return: A list of file IDs of the uploaded files on Google Drive.
    """
    # Authenticate and create the PyDrive client
    drive = authenticate_gdrive()

    # Ensure file_paths is a list
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    uploaded_file_ids = []

    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue

        # Create a file object and set its content from the local file
        file_name = os.path.basename(file_path)
        file = drive.CreateFile({'title': file_name, 'parents': [{'id': drive_folder_id}]})
        file.SetContentFile(file_path)

        try:
            # Upload the file to Google Drive
            file.Upload()
            uploaded_file_ids.append(file['id'])
            print(f"Uploaded {file_path} to Google Drive with file ID {file['id']}")

            # Delete the local file if specified
            if delete_local:
                os.remove(file_path)
                print(f"Deleted local file: {file_path}")
        except Exception as e:
            print(f"An error occurred while uploading {file_path}: {e}")

    return uploaded_file_ids


def download_from_drive(drive_file_id: str, local_path: str) -> bool:
    """
    Downloads a file from Google Drive using its file ID and saves it to the specified local path.

    :param drive_file_id: The ID of the file on Google Drive.
    :param local_path: The path where the file will be saved locally.
    :return: True if the file was successfully downloaded, False otherwise.
    """
    # Authenticate and create the PyDrive client
    drive = authenticate_gdrive()

    try:
        # Create a file object and set its ID
        file = drive.CreateFile({'id': drive_file_id})

        # Download the file content to the local path
        file.GetContentFile(local_path)
        print(f"Downloaded file ID {drive_file_id} to {local_path}")
        return True
    except Exception as e:
        print(f"An error occurred while downloading file ID {drive_file_id}: {e}")
        return False


def delete_drive_file(drive_file_id: str) -> bool:
    """
    Deletes a file on Google Drive given its file ID.

    :param drive_file_id: The ID of the file to delete on Google Drive.
    :return: True if the file was successfully deleted, False otherwise.
    """
    # Authenticate and create the PyDrive client
    drive = authenticate_gdrive()

    try:
        # Create a file object and set its ID
        file = drive.CreateFile({'id': drive_file_id})

        # Delete the file from Google Drive
        file.Delete()
        print(f"Successfully deleted Google Drive file ID: {drive_file_id}")
        return True
    except Exception as e:
        print(f"An error occurred while deleting file ID {drive_file_id} from Google Drive: {e}")
        return False


def delete_local_file(file_path: str) -> bool:
    """
    Deletes a local file given its file path.

    :param file_path: The path of the file to delete.
    :return: True if the file was successfully deleted, False otherwise.
    """
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Successfully deleted local file: {file_path}")
            return True
        else:
            print(f"File not found: {file_path}")
            return False
    except Exception as e:
        print(f"An error occurred while deleting the file {file_path}: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    # Define the local file paths and the destination folder ID on Google Drive
    file_paths = ['path/to/your/file1.txt', 'path/to/your/file2.txt']
    drive_folder_id = 'your_drive_folder_id'
    delete_local = True  # Set to True if you want to delete local files after upload

    # Call the function to upload files to Google Drive
    uploaded_file_ids = upload_to_drive(file_paths, drive_folder_id, delete_local)

    # Print the IDs of the uploaded files
    print("Uploaded file IDs:", uploaded_file_ids)

    # Define the file ID from Google Drive and the local path to save the file
    drive_file_id = 'your_drive_file_id'
    local_path = 'path/to/save/your/file.h5'

    # Call the function to download the file from Google Drive
    success = download_from_drive(drive_file_id, local_path)

    # Check if the file was successfully downloaded
    if success:
        print("File downloaded successfully.")
    else:
        print("File download failed.")
