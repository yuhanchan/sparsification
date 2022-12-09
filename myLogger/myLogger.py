import logging
import inspect
from os.path import basename
import sys

logging_level = logging.DEBUG
logger = None

def set_level(level):
    global logging_level
    if level == 'DEBUG':
        logging_level = logging.DEBUG
    elif level == 'INFO':
        logging_level = logging.INFO
    elif level == 'WARNING':
        logging_level = logging.WARNING
    elif level == 'ERROR':
        logging_level = logging.ERROR
    elif level == 'CRITICAL':
        logging_level = logging.CRITICAL
    else:
        print(f"{level} is not a valid logging level, please use one of the following: DEBUG, INFO, WARNING, ERROR, CRITICAL")
        sys.exit(1)

    
def get_logger(name):
    global logger
    logging.basicConfig(level=logging_level, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(name)

    
def debug(message):
    if logger is not None:
        logger.debug(f"{basename(inspect.stack()[1].filename)} - {message}")


def info(message):
    if logger is not None:
        logger.info(f"{basename(inspect.stack()[1].filename)} - {message}")
        print(f"{basename(inspect.stack()[1].filename)} - {message}")


def warning(message):
    if logger is not None:
        logger.warning(f"{basename(inspect.stack()[1].filename)} - {message}")
    

def error(message):
    if logger is not None:
        logger.error(f"{basename(inspect.stack()[1].filename)} - {message}")

        
def critical(message):
    if logger is not None:
        logger.critical(f"{basename(inspect.stack()[1].filename)} - {message}")