import os
import shutil
"""
Simple script for combining audio files from various directories into a single combined directory
with classes separated into individual directories, inferred from the last part of the original file
name, following an underscore(_)
"""
# Define categories you want to group
target_categories = ["bark", "growl", "grunt", "howl", "whine", "unknown"]

# directories to pull files from
sources = [
    "dataset/kaggle_dataset",
    "dataset/wav_25May2025",
    "dataset/wav_2July2025"
]

# Output directory for combined files
output_dir = "dataset/combined"

# Walk both sources
for src_root in sources:
    for root, _, files in os.walk(src_root):
        folder_name = os.path.basename(root)

        # match last part of folder name to one of the target categories
        for category in target_categories:
            if folder_name.lower().endswith(category):
                out_dir = os.path.join(output_dir, category)
                os.makedirs(out_dir, exist_ok=True)

                for file in files:
                    if file.lower().endswith(".wav"):
                        src_file = os.path.join(root, file)
                        dest_file = os.path.join(out_dir, f"{folder_name}_{file}")
                        shutil.copy2(src_file, dest_file)

print("All vocalization categories have been grouped.")
