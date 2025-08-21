import numpy as np
import vocalization_classifier.src.prep_data.preprocess as pp
import pytest

# test output shape when converting to spectrograms
@pytest.mark.unit
def test_audio_to_image_shape_and_range(sr):
    # create mock waveform
    t = np.arange(sr) / sr
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    img = pp.audio_to_image(x)  # convert to spectrogram

    # output from spectrogram should match configured settings
    assert img.shape == (pp.N_MELS, pp.TARGET_W, 1)  # should be configured shape
    assert np.isfinite(img).all()  # should be finite
    assert (img >= 0.0).all() and (img <= 1.0).all()  # should be normalized


# test edgecase of silent audio sample
@pytest.mark.unit
def test_audio_to_image_silent_edgecase(sr):
    # create mock silent signal
    x = np.zeros(sr, dtype=np.float32)
    img = pp.audio_to_image(x)  # convert to spectrogram

    # output should stay as all zeroes (silence) due to guard in preprocessing
    assert img.shape == (pp.N_MELS, pp.TARGET_W, 1)
    assert np.allclose(img, 0.0)
