"""
Used to clean up terminal output, can change the warning, error and info level of tensorflow logs
"""
import absl.logging
import warnings
import os

# set terminal log levels
def change_logging():
    absl.logging.set_verbosity('error')  
    warnings.filterwarnings("ignore") 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'