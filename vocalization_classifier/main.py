# allow project to be ran from any directory
import sys
from pathlib import Path

from src.checks.warning_level import change_logging

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

change_logging()  # limit logs that clog the terminal

# import libraries and files
from src.prep_data.get_df import build_dataframe
from vocalization_classifier.src.audio_classifier.build_model import create_and_train
from src.ui.cleanup import final_cleanup
from tf_lite_utils.tflite_utils import compare_models
from src.prep_data.get_split import get_train_val_split


"""
Main will load and split the dataset, create and train the full model (.keras), convert to tflite (.tflite),
compare the full and tflite models and then create a cpp file with a header for arduino integration
"""

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

final_cleanup()
print("Exiting...")
