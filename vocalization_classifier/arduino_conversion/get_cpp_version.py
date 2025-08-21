import shutil
import subprocess
import os
import re
from config import LITE_MODEL_PATH, MODEL_DIR

"""
Can be run as a standalone script or called functions to convert a tflite model (.tflite) to 
a new cpp file and header
Outputs to the "vocalization_classifier/models" directory
"""

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


# populate the c array and c header files
def create_content(content, arr_var_name):
    content = content.replace("unsigned char ", "const unsigned char ")
    content = content.replace("unsigned int ", "const unsigned int ")
    pat = rf"(const\s+unsigned\s+char\s+{re.escape(arr_var_name)}\s*\[\]\s*)="
    prolog = (
        "#if defined(ARDUINO_ARCH_AVR)\n"
        "#  include <avr/pgmspace.h>\n"
        "#  define AUDIO_MODEL_STORAGE PROGMEM\n"
        "#else\n"
        "#  define AUDIO_MODEL_STORAGE\n"
        "#endif\n\n"
    )
    repl = r"\1 AUDIO_MODEL_STORAGE alignas(16) ="
    content = prolog + re.sub(pat, repl, content, count=1)
    return content


# convert tflite model (.tflite) to cpp file and header
def convert_tflite_to_cpp():
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
    latest_dir_path = os.path.join(MODEL_DIR, "latest")  # check or create latest dir

    # verify version and latest directories
    os.makedirs(version_dir_path, exist_ok=True)
    os.makedirs(latest_dir_path, exist_ok=True)

    # paths for new files
    versioned_cpp_path = os.path.join(version_dir_path, "audio_classifier.cpp")
    versioned_h_path = os.path.join(version_dir_path, "audio_classifier.h")

    # get C array with xxd
    arr_var_name = "audio_classifier"
    result = subprocess.run(
        ["xxd", "-i", "-n", arr_var_name, LITE_MODEL_PATH],
        capture_output=True,
        text=True,
        check=True,
    )
    content = create_content(result.stdout, arr_var_name)
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

    print(
        f"\n\nModel array written to:\n  - {versioned_cpp_path}\n  - {versioned_h_path}"
    )

    # write latest copies to top level
    latest_cpp = os.path.join(latest_dir_path, "audio_classifier.cpp")
    latest_h = os.path.join(latest_dir_path, "audio_classifier.h")
    shutil.copy(versioned_cpp_path, latest_cpp)
    shutil.copy(versioned_h_path, latest_h)
    print(f"\nLatest version copied to:\n  - {latest_cpp}\n  - {latest_h}\n\n")


if __name__ == "__main__":
    convert_tflite_to_cpp() # gets tflite model from the models dir (should be only one)
