from io import BytesIO
from pathlib import Path

import pandas as pd


def read_dataframe(csv_path: str, csv_name: str) -> pd.DataFrame:
    """
    Reads a CSV file and returns a Pandas DataFrame.

    Args:
        csv_path (str): The path to the directory containing the CSV file.
        csv_name (str): The name of the CSV file.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the data from the CSV file.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the specified path.

    """
    # Check if the file exists
    full_path = csv_path + "/" + csv_name
    if not Path.exists(full_path):
        raise FileNotFoundError(f"The file {full_path} does not exist.")

    df = pd.read_csv(full_path)
    return df


def read_csv_from_bytes(csv_bytes: bytes) -> pd.DataFrame:
    """
    Reads a CSV file from bytes and returns a Pandas DataFrame.

    Args:
       csv_bytes (bytes): The CSV file in the bytes format.

    Returns:
       pd.DataFrame: A Pandas DataFrame containing the data from the CSV file.

    """
    df = pd.read_csv(BytesIO(csv_bytes))

    return df
