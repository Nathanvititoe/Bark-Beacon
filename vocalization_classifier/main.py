# allow project to be ran from any directory
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# import libraries and files
from src.prep_data.get_df import build_dataframe
from src.audio_classifier.build_model import create_and_train
from src.ui.cleanup import final_cleanup
from src.checks.warning_level import change_logging
from tf_lite_utils.tflite_utils import compare_models
from src.prep_data.get_split import get_train_val_split

change_logging()  # clean up terminal output

"""
Main will load and split the dataset, create and train the full model (.keras), convert to tflite (.tflite),
compare the full and tflite models and then create a cpp file with a header for arduino integration
"""

# REFACTOR
# boolean to toggle visuals
# modularized the dataset split and preprocess
# created config.py and integrated global config and directories throughout repo
# moved classifier logic all within build_model


# get dataframe from dataset
df = build_dataframe()

# setup the dataset
train_features, train_labels, val_features, val_labels = get_train_val_split(df)

# create, train and convert the model
classifier_history = create_and_train(
    train_features, train_labels, val_features, val_labels
)

# compare full model v tflite model accuracy
compare_models(val_features, val_labels)
print(val_features.min(), val_features.max())

final_cleanup()
print("Exiting...")
