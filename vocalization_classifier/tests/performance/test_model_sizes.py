import glob
import os
import pytest

MAX_TFLITE_BYTES = 512 * 1024

# get tf lite model
def _get_tflite():
    cands = glob.glob("**/*.tflite", recursive=True)
    return cands[0] if cands else None

# test final size of tflite model
@pytest.mark.performance
def test_tflite_size():
    path = _get_tflite()
    size = os.path.getsize(path)
    assert size <= MAX_TFLITE_BYTES, f"TFLite size: {size} bytes > maximum: {MAX_TFLITE_BYTES} bytes"
