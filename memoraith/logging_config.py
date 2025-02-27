import logging
import sys
from typing import Optional
from pathlib import Path

def setup_logging(log_level: int, log_file: Optional[str] = None, log_format: Optional[str] = None):
    """
    Configure logging for Memoraith with enhanced features.

    Args:
        log_level (int): The logging level to set
        log_file (str, optional): Path to the log file. If None, logs to console only.
        log_format (str, optional): Custom log format. If None, uses a default format.
    """
    logger = logging.getLogger('memoraith')
    logger.setLevel(log_level)

    # Use custom format if provided, otherwise use a default
    if log_format is None:
        log_format = '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file is provided)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Suppress logs from other libraries
    for lib in ['matplotlib', 'PIL', 'tensorflow', 'torch']:
        logging.getLogger(lib).setLevel(logging.WARNING)

    logger.info("Logging initialized")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name (str): Name of the logger

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(f"memoraith.{name}")

def log_exception(logger: logging.Logger, exc: Exception, level: int = logging.ERROR):
    """
    Log an exception with full traceback.

    Args:
        logger (logging.Logger): Logger instance
        exc (Exception): Exception to log
        level (int): Logging level for the exception
    """
    logger.log(level, "Exception occurred", exc_info=True)

def create_log_directory(base_path: str) -> str:
    """
    Create a directory for log files if it doesn't exist.

    Args:
        base_path (str): Base path for creating the log directory

    Returns:
        str: Path to the created log directory
    """
    log_dir = Path(base_path) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)

def set_log_level(logger: logging.Logger, level: int):
    """
    Set the logging level for a specific logger.

    Args:
        logger (logging.Logger): Logger instance
        level (int): Logging level to set
    """
    logger.setLevel(level)

def add_file_handler(logger: logging.Logger, file_path: str, level: int = logging.DEBUG):
    """
    Add a file handler to a logger.

    Args:
        logger (logging.Logger): Logger instance
        file_path (str): Path to the log file
        level (int): Logging level for the file handler
    """
    handler = logging.FileHandler(file_path)
    handler.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
