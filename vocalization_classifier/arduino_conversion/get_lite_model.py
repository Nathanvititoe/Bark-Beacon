from sklearn.model_selection import train_test_split
from tf_lite_utils.converter.tflite_converter import (
    convert_for_microcontroller,
    analyze_tflite_model,
)
from src.prep_data.get_df import build_dataframe
from vocalization_classifier.src.prep_data.preprocess.preprocess import load_data

"""
 This is meant to be a standalone script separate from main, that can be run to create a tf lite file (.tflite)
 from an existing full model (.keras) file
"""


# get validation data
df = build_dataframe()
train_data, val_data = train_test_split(
    df, test_size=0.1, stratify=df["classID"], random_state=42
)
val_features, val_labels = load_data(val_data, df_type="validation")

convert_for_microcontroller()
print("Analyzing TFLite Model specs...")
analyze_tflite_model()
