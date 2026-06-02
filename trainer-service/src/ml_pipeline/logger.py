import logging
from pathlib import Path


def get_logger(
    file_path: str = "logs",
    file_name: str = "ml_pipeline.log",
    level: int = logging.INFO,
):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    filename = Path().joinpath(file_path, file_name)

    Path(file_path).mkdir(parents=True, exist_ok=True)
    Path(filename).touch(exist_ok=True)

    if filename is None:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Консольный вывод (надо?)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(filename, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
