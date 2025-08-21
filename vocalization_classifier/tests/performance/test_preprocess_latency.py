import time
import numpy as np
import pytest
import vocalization_classifier.src.prep_data.preprocess as pp

MAX_SPEC_MS = 50  # each image should convert in under 1.5s


# test latency for conversion from audio to spectrogram image
@pytest.mark.performance
def test_audio_to_image_latency(sr, duration):
    target_len = int(sr * duration)
    # create random noise
    x = np.random.default_rng(0).standard_normal(target_len).astype(np.float32) * 0.05

    # run once beforehand so that init/setup isnt timed
    _ = pp.audio_to_image(x)

    start = time.perf_counter()  # start timer
    _ = pp.audio_to_image(x)  # convert to image
    time_total_ms = (time.perf_counter() - start) * 1000.0

    # ensure latency is less than the max latency configured above
    assert (
        time_total_ms <= MAX_SPEC_MS
    ), f"Spectrogram conversion latency: {time_total_ms:.1f} ms > {MAX_SPEC_MS:.1f} ms"
