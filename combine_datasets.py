import os
import shutil

# Define categories you want to group
target_categories = ["bark", "growl", "grunt", "howl", "whine"]

# Source roots
sources = [
    "dataset/kaggle_dataset",
    "dataset/wav_25May2025",
    "dataset/wav_2July2025",
    
]

# Output root
output_root = "dataset/combined"

# Walk both sources
for src_root in sources:
    for root, _, files in os.walk(src_root):
        folder_name = os.path.basename(root)

        # Try to match last part of folder name to one of the target categories
        for category in target_categories:
            if folder_name.lower().endswith(category):
                out_dir = os.path.join(output_root, category)
                os.makedirs(out_dir, exist_ok=True)

                for file in files:
                    if file.lower().endswith(".wav"):
                        src_file = os.path.join(root, file)
                        dest_file = os.path.join(out_dir, f"{folder_name}_{file}")
                        shutil.copy2(src_file, dest_file)

print("✅ All target categories have been grouped.")
