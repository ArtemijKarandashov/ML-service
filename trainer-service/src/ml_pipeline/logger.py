import logging
from pathlib import Path


def get_logger(
    file_path: str = "logs",
    file_name: str = "ml-pipeline.log",
    level: int = logging.INFO,
):
    """
    Create and configure a logger with console and file handlers.

    Args:
        file_path (str): The directory path where the log file will be stored. Defaults to "logs".
        file_name (str): The name of the log file. Defaults to "ml_pipeline.log".
        level (int): The logging level for the logger. Defaults to logging.INFO.

    Returns:
        logging.Logger: A configured logger instance with console and file handlers.

    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    filename = Path().joinpath(file_path, file_name)

    Path(file_path).mkdir(parents=True, exist_ok=True)

    if filename is None:
        return logger

    Path(filename).touch(exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(filename, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
