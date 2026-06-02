import pandas as pd

# Dictionary with the encoding strategies for categorical features
encoding_strategy = {
    "person_gender": lambda df, column: encode_binary(
        df,
        column,
        "male",
    ),
    "previous_loan_defaults_on_file": lambda df, column: encode_binary(
        df,
        column,
        "yes",
    ),
    "person_education": lambda df, column: encode_one_hot(df, column),
    "person_home_ownership": lambda df, column: encode_one_hot(df, column),
    "loan_intent": lambda df, column: encode_one_hot(df, column),
}


def encode_binary(df: pd.DataFrame, column: str, target_value: str) -> pd.DataFrame:
    """
    Encodes a binary categorical feature into an integer value.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to be encoded.
        target_value (str): The target value for the encoding.

    Returns:
        pd.DataFrame: A new DataFrame with the encoded column appended.

    """
    df_encoded = df[column].apply(
        lambda value: 1 if value.lower() == target_value else 0,
    )
    df_encoded.rename(str(column + "_encoded"), inplace=True)

    return df_encoded


def encode_one_hot(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df_encoded = (
        pd.get_dummies(
            df[column].apply(pd.Series).stack(),
        )
        .groupby(level=0)
        .sum()
    )

    return df_encoded


def encode_columns(df_sanitized: pd.DataFrame, encoding_strategy: dict) -> pd.DataFrame:
    df_encoded = df_sanitized.copy()

    for column in df_encoded:
        if column in encoding_strategy:
            encoded_column = encoding_strategy[column](df_encoded, column)
            df_encoded = pd.concat(
                [df_encoded.drop(column, axis=1), encoded_column],
                axis=1,
            )

    return df_encoded
