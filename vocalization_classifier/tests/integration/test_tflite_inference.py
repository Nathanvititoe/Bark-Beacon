import os
import glob
import numpy as np
import pytest
import src.prep_data.preprocess as pp
from ai_edge_litert.interpreter import Interpreter


# get a tflite model from the models directory
def _get_tflite():
    cands = glob.glob("**/*.tflite", recursive=True)
    return cands[0] if cands else None


# check spectrogram shape matches models expected input
def _verify_spec_shape(spec, input_detail):
    want = tuple(input_detail["shape"])
    have = spec.shape

    if have == want:
        return spec

    if len(have) == 4 and len(want) == 4:
        swapped = np.transpose(spec, (0, 2, 1, 3))
        if swapped.shape == want:
            return swapped
    raise AssertionError(
        f"feature shape {have} doesn't match model input {want} after preprocessing."
    )


# test full pipeline and inference using tflite model on file from test_audio dir
@pytest.mark.integration
def test_full_pipeline():
    model = _get_tflite()  # get tf lite model
    if not model:
        pytest.skip("no .tflite model found")

    # get wav sample from test audio files
    root = pp.AUDIO_ROOT_PATH
    wavs = glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
    if not wavs:
        pytest.skip("No wave files in test audio dir")

    # load file and convert to spectrogram
    spec = pp.audio_to_image(pp.load_file(wavs[0]))[np.newaxis, ...]

    # load interpreter and get input/output details
    itp = Interpreter(model_path=model)
    itp.allocate_tensors()
    inputs = itp.get_input_details()
    outputs = itp.get_output_details()

    # Verify input/output tensors
    assert (
        len(inputs) == 1
    ), f"Only 1 input tensor expected, there were {len(inputs)} input tensors"
    assert len(outputs) >= 1, "Output must contain at least one output tensor"

    inp = inputs[0]
    out = outputs[0]

    # ensure shape matches expected
    data = _verify_spec_shape(spec, inp)

    # check output shape to model input shape
    assert data.shape == tuple(
        inp["shape"]
    ), f"input shape: {data.shape} does not match model input shape: {tuple(inp['shape'])}"

    # read output and check batch
    itp.set_tensor(inp["index"], data)
    itp.invoke()
    y = itp.get_tensor(out["index"])
    assert y.ndim == 2 and y.shape[0] == 1, f"unexpected output shape {y.shape}"
    if y.dtype == np.float32:
        assert np.isfinite(y).all()  # check finite value
        assert (y >= -1e-3).all() and (y <= 1 + 1e-3).all()  # check normalized values
        assert y.shape[1] == 5  # bark/growl/howl/whine/unknown
