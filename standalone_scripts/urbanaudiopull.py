import os
import shutil
import pandas as pd

# CONFIG
URBAN_CSV = "/home/nathanvititoe/CSC370/Assignment5.1/dataset/UrbanSound8K.csv"   # path to your UrbanSound8K metadata CSV
URBAN_AUDIO_ROOT = "/home/nathanvititoe/CSC370/Assignment5.1/dataset/dataset_folds"  # root folder containing fold1 ... fold10
OUTPUT_UNKNOWN_DIR = "./unknown"  # where to copy the unknown samples
NUM_UNKNOWN_SAMPLES = 600  # approx number of unknown samples to copy

# Load metadata
urban_df = pd.read_csv(URBAN_CSV)

# Filter out dog_bark samples (we only want non-dog classes)
unknown_df = urban_df[urban_df["class"] != "dog_bark"]

# Get unique non-dog classes
unique_classes = unknown_df["class"].unique()

# Calculate how many samples to pull per class
samples_per_class = max(1, NUM_UNKNOWN_SAMPLES // len(unique_classes))

# Sample evenly from each class
balanced_unknown_samples = (
    unknown_df.groupby("class", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), samples_per_class), random_state=42))
    .reset_index(drop=True)
)

print(f"Pulling {len(balanced_unknown_samples)} samples into '{OUTPUT_UNKNOWN_DIR}'...")

# Create output directory
os.makedirs(OUTPUT_UNKNOWN_DIR, exist_ok=True)

# Copy each file into the unknown directory
for _, row in balanced_unknown_samples.iterrows():
    fold = f"fold{row['fold']}"
    src = os.path.join(URBAN_AUDIO_ROOT, fold, row["slice_file_name"])
    dst = os.path.join(OUTPUT_UNKNOWN_DIR, row["slice_file_name"])
    try:
        shutil.copy(src, dst)
    except Exception as e:
        print(f"Failed to copy {src}: {e}")

print("✅ Unknown samples copied successfully.")
