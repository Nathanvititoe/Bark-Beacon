import numpy as np
import src.prep_data.preprocess as pp


# test preprocessing to create uniform duration, making all files 4s by padding or trimming
# every file should return with the set length/duration
def test_duration_conversion(write_wav, sr, duration):
    target_len = int(sr * duration)

    # test samples that are too long and need trimmed
    # create mock sample thats 1.5x the length needed
    long_sample = np.ones(int(target_len * 1.5), dtype=np.float32)
    f_long = write_wav("Dog/long.wav", long_sample, sr)
    y_long = pp.load_file(f_long)

    # output from load_file should create samples that match target
    assert y_long.shape == (target_len,)
    assert np.allclose(y_long, 1.0)

    # test samples that are shorter than set length/duration
    short_sample = np.ones(int(target_len * 0.25), dtype=np.float32)
    f_short = write_wav("Dog/short.wav", short_sample, sr)
    y_short = pp.load_file(f_short)
    # output from load_file should create samples that match target
    assert y_short.shape == (target_len,)
    cut = int(0.25 * target_len)
    assert np.allclose(y_short[:cut], 1.0)
    assert np.allclose(y_short[cut:], 0.0)


# test preprocessing to convert all files to mono channel
def test_channel_conversion(write_wav, sr, duration):
    target_len = int(sr * duration)

    # create mock left/right channel sample
    left = np.ones(target_len, dtype=np.float32)
    right = np.zeros(target_len, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)  # combine them for dual channel

    f = write_wav("Dog/stereo.wav", stereo, sr)
    y = pp.load_file(f)

    # output from load_file should be all monochannel
    assert y.ndim == 1 and y.shape[0] == target_len
    assert np.allclose(y, 0.5, atol=1e-6)


# # test preprocessing to convert all files to uniform sampling rate
def test_resampling(write_wav, sr, duration):
    target_len = int(sr * duration)

    # create mock sample at too low of a sampling rate
    alt_sr = 8000
    t = np.arange(int(alt_sr * duration)) / alt_sr
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    f = write_wav("Dog/lowrate.wav", x, alt_sr)
    y = pp.load_file(f)

    # output should match target length (sr * duration) bc of resampling
    assert y.shape == (target_len,)
    assert np.isclose(np.mean(y), 0.0, atol=1e-2)  # avg should be near 0

    # create mock sample at too high of a sampling rate
    alt_sr2 = 22050
    t2 = np.arange(int(alt_sr2 * duration)) / alt_sr2
    x2 = np.sin(2 * np.pi * 440 * t2).astype(np.float32)

    f2 = write_wav("Dog/highrate.wav", x2, alt_sr2)
    y2 = pp.load_file(f2)

    # output should match target length (sr * duration) bc of resampling
    assert y2.shape == (target_len,)
