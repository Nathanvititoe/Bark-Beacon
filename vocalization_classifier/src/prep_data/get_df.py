import pandas as pd
import os
from src.prep_data.evaluate_dataset import plot_dataset
from config import AUDIO_ROOT_PATH, SHOW_VISUALS

"""
 this converts the dataset to a pandas dataframe that can be used by tensorflow for model creation  
"""


def build_dataframe():
    rows = []
    class_map = {}  # map class name to ID
    class_id = 0

    for class_name in sorted(os.listdir(AUDIO_ROOT_PATH)):
        class_dir = os.path.join(AUDIO_ROOT_PATH, class_name)
        if not os.path.isdir(class_dir):
            continue

        if class_name not in class_map:
            class_map[class_name] = class_id
            class_id += 1

        for fname in os.listdir(class_dir):
            if fname.lower().endswith(".wav"):
                rows.append(
                    {
                        "slice_file_name": fname,
                        "class": class_name,
                        "classID": class_map[class_name],
                    }
                )

    df = pd.DataFrame(rows)
    if SHOW_VISUALS:
        plot_dataset(df)

    return df
