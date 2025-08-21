import os
from pathlib import Path

# terminal output '1','2', or '3' (3 is least logs)
LOG_LEVEL = "3"

# root folder of the project
PROJECT_ROOT = Path(__file__).resolve().parent  # Repo Root

# Directory paths
DATASET_DIR = PROJECT_ROOT / "dataset" / "combined"
BASE_LOGIC_DIR = os.path.dirname(
    os.path.abspath(__file__) # vocalization classifier dir
)  
MODEL_DIR = os.path.join(BASE_LOGIC_DIR, "models")

FULL_MODEL_PATH = os.path.join(MODEL_DIR, "BarkBeacon_Full.keras")
LITE_MODEL_PATH = os.path.join(MODEL_DIR, "BarkBeacon_Lite.tflite")
AUDIO_ROOT_PATH = os.getenv("AUDIO_ROOT_PATH", str(DATASET_DIR))
TEST_AUDIO_DIR = os.path.join(BASE_LOGIC_DIR, "test_audio")

# Model config options
SHOW_VISUALS = False
USER_PREDICT = False
VALID_SPLIT = 0.1  # % of dataset to use for validation
BATCH_SIZE = 16
SAMPLE_RATE = 16000
DURATION_SEC = 4
NUM_EPOCHS = 100

# Model Characteristics
NUM_CLASSES = 5
CLASS_NAMES = ["bark", "growl", "whine", "howl", "unknown"]

# Spectrogram configuration
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

# time configuration for spectrograms
TARGET_W = 256
TARGET_H = N_MELS
