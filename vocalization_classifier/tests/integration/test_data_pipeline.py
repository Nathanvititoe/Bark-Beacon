import numpy as np
import pandas as pd
import pytest
import src.prep_data.preprocess as pp

# test the data pipeline to ensure audio files make it through the whole process and 
# come out with the correct shape to match the model input
@pytest.mark.integration
def test_data_pipeline(write_wav, sr, duration, monkeypatch):

    # create two small mock audio files (.wav)
    target_len = int(sr * duration)
    bark = np.ones(target_len, dtype=np.float32)
    growl = 0.1 * np.ones(target_len, dtype=np.float32)

    # put them into subfolders so their classes are inferred
    write_wav("bark/bark_01.wav", bark, sr)
    write_wav("growl/growl_01.wav", growl, sr)

    # use the temporary audio root for this test
    monkeypatch.setattr(pp, "AUDIO_ROOT_PATH", write_wav.base, raising=False)

    # create dataframe from mock files
    df = pd.DataFrame(
        [
            {"class": "bark", "classID": 0, "slice_file_name": "bark_01.wav"},
            {"class": "growl", "classID": 1, "slice_file_name": "growl_01.wav"},
        ]
    )

    # run pipeline
    X, y = pp.load_data(df, df_type="training")

    # files should be output in the correct shape, duration, sampling rate, and mono channel
    assert X.shape[0] == 2
    assert X.shape[1:] == (pp.N_MELS, pp.TARGET_W, 1)
    assert y.shape == (2,)
    assert set(y.tolist()) == {0, 1}
