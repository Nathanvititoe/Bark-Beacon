import shutil
import subprocess
import os

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

FULL_MODEL_PATH = os.path.join(MODEL_DIR, "BarkBeacon_Full.h5")
LITE_MODEL_PATH = os.path.join(MODEL_DIR, "BarkBeacon_Lite.tflite")
C_ARR_PATH = os.path.join(MODEL_DIR, "BarkBeacon_c.cc")

# convert tf lite model to c array for arduino
def convert_tflite_to_c_array():
    if not os.path.exists(LITE_MODEL_PATH):
        raise FileNotFoundError(f"TFLite model not found at: {LITE_MODEL_PATH}")

    if not shutil.which("xxd"):
        raise EnvironmentError("xxd not found. Install it (e.g., sudo apt install xxd)")

    os.makedirs(os.path.dirname(C_ARR_PATH), exist_ok=True)

    try:
        with open(C_ARR_PATH, "w") as out_file:
            subprocess.run(["xxd", "-i", LITE_MODEL_PATH], check=True, stdout=out_file)
        print(f"Successfully wrote C array to {C_ARR_PATH}")
    except subprocess.CalledProcessError as e:
        print(f"xxd failed with error: {e}")

if __name__ == "__main__":
    convert_tflite_to_c_array()
