import os
from sklearn.model_selection import train_test_split
from tf_lite_utils.converter.tflite_converter import convert_for_microcontroller, get_representative_dataset, analyze_tflite_model
from src.prep_data.get_df import build_dataframe
from src.prep_data.preprocess import load_data
"""
 This is meant to be a standalone script separate from main, that can be run to create a tf lite file (.tflite)
 from an existing full model (.keras) file
"""

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "./vocalization_classifier/models")

FULL_MODEL_PATH = os.path.join(MODEL_DIR, "BarkBeacon_Full.keras")
LITE_MODEL_PATH = os.path.join(MODEL_DIR, "BarkBeacon_Lite.tflite")
AUDIO_ROOT_PATH = '../dataset/combined'

# preprocess config
SAMPLE_RATE = 16000 # sample rate to downsample to
DURATION_SEC = 4 # time length of audio file (seconds)

# get validation data
df = build_dataframe(AUDIO_ROOT_PATH)
train_data, val_data = train_test_split(df, test_size=0.1, stratify=df['classID'], random_state=42)
val_features, val_labels = load_data(AUDIO_ROOT_PATH, val_data, SAMPLE_RATE, DURATION_SEC, df_type="validation")

# get rep dataset
rep_dataset = get_representative_dataset(val_features)
convert_for_microcontroller(FULL_MODEL_PATH, LITE_MODEL_PATH, rep_dataset)
print("Analyzing TFLite Model specs...")
analyze_tflite_model(LITE_MODEL_PATH)