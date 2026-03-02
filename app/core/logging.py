import logging
import os


def setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=date_fmt,
        handlers=[logging.StreamHandler()],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
