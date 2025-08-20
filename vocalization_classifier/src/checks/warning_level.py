"""
Used to clean up terminal output, can change the warning, error and info level of tensorflow logs
"""

# https://github.com/tensorflow/tensorflow/issues/73487
import warnings
import os
import tensorflow as tf
import logging
# from config import LOG_LEVEL


# set terminal log levels
def change_logging():
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)
