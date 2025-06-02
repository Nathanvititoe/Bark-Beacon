import pandas as pd
import os

def build_dataframe(combined_root):
    rows = []
    class_map = {}  # map class name to ID
    class_id = 0

    for class_name in sorted(os.listdir(combined_root)):
        class_dir = os.path.join(combined_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        if class_name not in class_map:
            class_map[class_name] = class_id
            class_id += 1

        for fname in os.listdir(class_dir):
            if fname.lower().endswith(".wav"):
                rows.append({
                    "slice_file_name": fname,
                    "class": class_name,
                    "classID": class_map[class_name]
                })

    df = pd.DataFrame(rows)
    return df
