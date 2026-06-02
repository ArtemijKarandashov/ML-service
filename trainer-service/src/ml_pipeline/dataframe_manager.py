import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df, target_column: str, test_size: float = 0.1) -> tuple:
    """
    Split data into training and testing sets based on the target column.

    Args:
        df (pd.DataFrame): The input dataframe containing the features and target.
        target_column (str): The name of the target column to separate from the features.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.1.

    Returns:
        tuple (x_train, x_test, y_train, y_test): A tuple containing four pandas Series/DataFrames:
        x_train (pd.DataFrame): Features for the training set.
        x_test (pd.DataFrame): Features for the testing set.
        y_train (pd.Series): Target values for the training set.
        y_test (pd.Series): Target values for the testing set.

    """
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(target_column, axis=1),
        df[target_column],
        test_size=test_size,
        random_state=0,
    )

    return (x_train, x_test, y_train, y_test)


def balance_dateframe(
    df: pd.DataFrame,
    target_column: str,
    cutout_threshold: float = 0.65,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Balance a dataframe by down-sampling the majority class to match the minority class size.

    Args:
        df (pd.DataFrame): The input dataframe containing features and target values.
        target_column (str): The name of the column indicating the class labels (0 or 1).
        cutout_threshold (float, optional): Threshold for extreme skewness; if exceeded, returns original data. Defaults to 0.65.
        random_state (int, optional): Seed for reproducibility in sampling process. Defaults to 0.

    Returns:
        pd.DataFrame: A balanced dataframe containing only the minority class and a downsampled majority class, or the original dataframe if skewness is extreme.

    Raises:
        ValueError: If target_column does not exist in df.

    """
    if target_column not in df.columns:
        msg = f"Column {target_column} does not exist in the DataFrame."
        raise ValueError(msg)

    df_zeros = df[df[target_column] == 0]
    df_ones = df[df[target_column] == 1]

    df_minority, df_majoruty = df_zeros, df_ones
    if len(df_ones) <= len(df_zeros):
        df_minority, df_majoruty = df_ones, df_zeros

    delta = len(df_majoruty) - len(df_minority)
    # if skew is extreme (difference is greater than threshold) return original dataframe (data is garbage anyway)
    if delta / len(df) >= cutout_threshold:
        return df

    df_majority_downsampled = df_majoruty.sample(
        n=len(df_minority),
        random_state=random_state,
    )
    df_balanced = pd.concat([df_majority_downsampled, df_minority], axis=0)

    return df_balanced


# Unused
def drop_low_corr_columns(
    df: pd.DataFrame,
    target_column: str,
    drop_threshold: float = 0.004,
) -> pd.DataFrame:
    """
    Drop low-correlation columns from the dataframe based on their correlation with the target column.

    Args:
        df (pd.DataFrame): The input dataframe containing features and target values.
        target_column (str): The name of the target column to use for computing correlations.
        drop_threshold (float, optional): Threshold below which a column will be dropped; defaults to 0.004 (0.4%).

    Returns:
        pd.DataFrame: A new dataframe with low-correlation columns removed.

    """
    df_cleaned = df.copy()
    correlation_series = df.corrwith(df[target_column]).abs()
    for column in df.columns:
        if correlation_series[column] < drop_threshold:
            print(
                f"Dropper column {column} with correlation {correlation_series[column]}",
            )
            df_cleaned.drop(column, axis=1)

    return df_cleaned
