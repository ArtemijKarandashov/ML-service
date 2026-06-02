from io import BytesIO

import pandas as pd


def read_dataframe(csv_path: str, csv_name: str) -> pd.DataFrame:
    # TODO:
    # checks:
    #   - file exists?
    #   - is it csv?
    df = pd.read_csv(csv_path + "/" + csv_name)

    return df


def read_csv_from_bytes(csv_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(csv_bytes))

    return df
