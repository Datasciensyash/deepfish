import logging

from deepfish.constants import LOGS_DIR


def create_logger(name: str, need_logging: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    if need_logging:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s: %(name)s: %(levelname)s: %(message)s"))
    logger.addHandler(handler)

    handler = logging.FileHandler(LOGS_DIR / f"{name.lower()}.txt")
    handler.setFormatter(logging.Formatter("%(asctime)s: %(name)s: %(levelname)s: %(message)s"))
    logger.addHandler(handler)

    return logger
