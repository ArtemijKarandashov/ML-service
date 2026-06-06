import pandas as pd

training_allowed_columns = [
    "loan_status",
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "person_gender_encoded",
    "Associate",
    "Bachelor",
    "Master",
    "MORTGAGE",
    "OTHER",
    "OWN",
    "RENT",
    "DEBTCONSOLIDATION",
    "EDUCATION",
    "HOMEIMPROVEMENT",
    "MEDICAL",
    "PERSONAL",
    "VENTURE",
    "previous_loan_defaults_on_file_encoded",
]

prediction_allowed_columns = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "person_gender_encoded",
    "Associate",
    "Bachelor",
    "Master",
    "MORTGAGE",
    "OTHER",
    "OWN",
    "RENT",
    "DEBTCONSOLIDATION",
    "EDUCATION",
    "HOMEIMPROVEMENT",
    "MEDICAL",
    "PERSONAL",
    "VENTURE",
    "previous_loan_defaults_on_file_encoded",
]


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
    # If value is target_value, encode as 1; otherwise 0
    df_encoded = df[column].apply(
        lambda value: 1 if value.lower() == target_value else 0,
    )
    df_encoded.rename(str(column + "_encoded"), inplace=True)

    return df_encoded


def encode_one_hot(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Encodes a categorical feature into one-hot encoded columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to be encoded.

    Returns:
        pd.DataFrame: A new DataFrame with the one-hot encoded columns appended.

    """
    # Create a sparse matrix with N columns, each representing one unique values
    # The encoded value is set to 1, all other are set to 0
    df_encoded = (
        pd.get_dummies(
            df[column].apply(pd.Series).stack(),
        )
        .groupby(level=0)
        .sum()
    )

    return df_encoded


def encode_columns(df_sanitized: pd.DataFrame, encoding_strategy: dict) -> pd.DataFrame:
    """
    Applies the specified encoding strategies to a sanitized DataFrame.

    Args:
        df_sanitized (pd.DataFrame): The input DataFrame with no missing values and all columns already processed for non-numeric data types.
        encoding_strategy (dict): A dictionary mapping column names to their respective encoding functions.\
            Each value in the dictionary should be a callable that takes a DataFrame and a column name as arguments\
            and returns a DataFrame containing the encoded column(s).

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns removed and replaced with their encoded versions.\
            If a column is not found in the encoding_strategy, it remains unchanged in the returned DataFrame.

    """
    df_encoded = df_sanitized.copy()

    for column in df_encoded:
        if column in encoding_strategy:
            encoded_column = encoding_strategy[column](df_encoded, column)
            df_encoded = pd.concat(
                [df_encoded.drop(column, axis=1), encoded_column],
                axis=1,
            )

    return df_encoded
