import logging


def setup_logger(level: int = logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
