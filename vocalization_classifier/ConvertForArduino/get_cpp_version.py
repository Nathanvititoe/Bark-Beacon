import shutil
import subprocess
import os
import re

"""
Can be run as a standalone script or called functions to convert a tflite model (.tflite) to 
a new cpp file and header
Outputs to the "vocalization_classifier/models" directory
"""
# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../models")
FULL_MODEL_PATH = os.path.join(MODEL_DIR, "BarkBeacon_Full.keras")

# get version num for this creation
def get_next_version():
    pattern = re.compile(r"audio_classifier_v(\d+)")
    max_version = 0
    for name in os.listdir(MODEL_DIR):
        match = pattern.match(name)
        if match:
            version = int(match.group(1))
            max_version = max(max_version, version)
    return max_version + 1

# convert tflite model (.tflite) to cpp file and header
def convert_tflite_to_cpp(LITE_MODEL_PATH):
    # throw err if lite model cant be found
    if not os.path.exists(LITE_MODEL_PATH):
        raise FileNotFoundError(f"TFLite model not found: {LITE_MODEL_PATH}")
    
    # throw err if xxd doesnt exist
    if not shutil.which("xxd"):
        raise EnvironmentError("xxd not found. Install it (sudo apt install xxd)")

    # get version num and create dir for it
    version = get_next_version()
    version_dir_name = f"audio_classifier_v{version}"
    version_dir_path = os.path.join(MODEL_DIR, version_dir_name)
    latest_dir_path = os.path.join(MODEL_DIR, "latest") # check or create latest dir

    # verify version and latest directories
    os.makedirs(version_dir_path, exist_ok=True)
    os.makedirs(latest_dir_path, exist_ok=True)
    
    # paths for new files
    versioned_cpp_path = os.path.join(version_dir_path, "audio_classifier.cpp")
    versioned_h_path = os.path.join(version_dir_path, "audio_classifier.h")

    # get C array with xxd
    result = subprocess.run(["xxd", "-i", LITE_MODEL_PATH], capture_output=True, text=True, check=True)
    content = result.stdout
    content = content.replace("unsigned char ", "const unsigned char ")
    content = content.replace("[]", "audio_classifier[]")
    content = content.replace("_len", "audio_classifier_len")
    content = content.replace("unsigned int ", "const unsigned int ")
    content = f'#include "audio_classifier.h"\n\n{content}\n'

    # write cpp file
    with open(versioned_cpp_path, "w") as f:
        f.write(content)

    # write matching header(.h) file
    include_guard = f"AUDIO_CLASSIFIER_V{version}_H"
    with open(versioned_h_path, "w") as f:
        f.write(f"#ifndef {include_guard}\n#define {include_guard}\n\n")
        f.write("extern const unsigned char audio_classifier[];\n")
        f.write("extern const unsigned int audio_classifier_len;\n\n")
        f.write(f"#endif // {include_guard}\n")

    print(f"Model array written to:\n  - {versioned_cpp_path}\n  - {versioned_h_path}")
    print(f"Version #: {version}")

    # write latest copies to top level
    latest_cpp = os.path.join(MODEL_DIR, "audio_classifier.cpp")
    latest_h = os.path.join(MODEL_DIR, "audio_classifier.h")
    shutil.copy(versioned_cpp_path, latest_cpp)
    shutil.copy(versioned_h_path, latest_h)
    print(f"Latest version copied to:\n  - {latest_cpp}\n  - {latest_h}")

if __name__ == "__main__":
    tf_lite_model = os.path.join(MODEL_DIR, "BarkBeacon_Lite.tflite")
    convert_tflite_to_cpp(tf_lite_model)