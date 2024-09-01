import os
import logging
from config import LOG_DIRECTORY, PROCESS_LOG_FILE, ERROR_LOG_FILE

def setup_logging():
    """Setup logging configuration and prepare directories."""
    os.makedirs(LOG_DIRECTORY, exist_ok=True)

    process_log_path = os.path.join(LOG_DIRECTORY, PROCESS_LOG_FILE)
    error_log_path = os.path.join(LOG_DIRECTORY, ERROR_LOG_FILE)

    error_logger = logging.getLogger('error_logger')
    error_logger.setLevel(logging.ERROR)
    error_handler = logging.FileHandler(error_log_path)
    error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    error_logger.addHandler(error_handler)

    process_logger = logging.getLogger('process_logger')
    process_logger.setLevel(logging.INFO)
    process_handler = logging.FileHandler(process_log_path)
    process_handler.setFormatter(logging.Formatter('%(message)s'))
    process_logger.addHandler(process_handler)

    return process_logger, error_logger
