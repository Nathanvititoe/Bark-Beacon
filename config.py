import os

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

FULL_MODEL_PATH = os.path.join(MODEL_DIR, "BarkBeacon_Full.keras")
LITE_MODEL_PATH = os.path.join(MODEL_DIR, "BarkBeacon_Lite.tflite")
AUDIO_ROOT_PATH = "../dataset/combined"
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
