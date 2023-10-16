import os
from logging.handlers import RotatingFileHandler
from logging import Formatter, getLogger, INFO, StreamHandler

WORKER_COUNT: int = 16
LOG_FORMAT: str = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
DATE_FTM: str = "%d.%m.%Y %H:%M"
MAX_FILE_SIZE: int = 1024 * 10024  # 1 МБ
MAX_FILES: int = 3


def get_file_handler(name: str) -> RotatingFileHandler:
    log_dir_name: str = "logging"
    if not os.path.exists(log_dir_name):
        os.mkdir(log_dir_name)
    file_handler: RotatingFileHandler = RotatingFileHandler(
        f"{log_dir_name}/{name}.log",
        maxBytes=MAX_FILE_SIZE,
        backupCount=MAX_FILES
    )
    file_handler.setFormatter(Formatter(LOG_FORMAT, datefmt=DATE_FTM))
    return file_handler


def get_stream_handler():
    stream_handler: StreamHandler = StreamHandler()
    stream_handler.setFormatter(Formatter(LOG_FORMAT))
    return stream_handler


def get_logger(name: str) -> getLogger:
    logger: getLogger = getLogger(name)
    logger.addHandler(get_file_handler(name))
    logger.addHandler(get_stream_handler())
    logger.setLevel(INFO)
    return logger
