import os
import numpy as np
import scipy.signal
import soundfile as sf
from tqdm import tqdm
import librosa
from config import (
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    TARGET_W,
    SAMPLE_RATE,
    DURATION_SEC,
    AUDIO_ROOT_PATH,
    SHOW_VISUALS,
)

"""
Test file to see if we can achieve similar accuracy without using YAMNet, but still 
converting audio samples to spectrograms and then running CNN image classification on them
"""
# from src.ui.visualization import plot_spectrograms, plot_waveform
from config import SAMPLE_RATE, DURATION_SEC, AUDIO_ROOT_PATH, SHOW_VISUALS


# Load audio, convert to uniform channel, sample rate and duration and return a float 32 numpy array
def load_file(filepath: str) -> np.ndarray:
    target_len = SAMPLE_RATE * DURATION_SEC
    audio, sr = sf.read(filepath, always_2d=False)  # read audio sample file

    # convert all files to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # resample to 16KHz
    if sr != SAMPLE_RATE:
        num_samples = int(SAMPLE_RATE * len(audio) / sr)
        audio = scipy.signal.resample(audio, num_samples)

    # pad/trim
    if len(audio) < target_len:
        pad = target_len - len(audio)
        audio = np.pad(audio, (0, pad), mode="constant")
    else:
        audio = audio[:target_len]

    return audio.astype(np.float32)


# get the file path from dataset
def get_file_path(row) -> str:
    c1 = os.path.join(AUDIO_ROOT_PATH, row.get("class", ""), row["slice_file_name"])
    c2 = os.path.join(AUDIO_ROOT_PATH, row["slice_file_name"])
    if os.path.exists(c1):
        return c1
    if os.path.exists(c2):
        return c2
    raise FileNotFoundError(f"Audio file not found: {c1} | {c2}")


# ------------------ audio -> image ------------------
def audio_to_image(waveform: np.ndarray) -> np.ndarray:
    spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
        center=False,
    )  # (n_mels, frames)

    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)  # (n_mels, frames)

    # normalize each file to [0,1]
    vmin, vmax = np.percentile(log_spectrogram, 5), np.percentile(log_spectrogram, 95)

    if vmax <= vmin:
        vmax = vmin + 1.0
    log_spectrogram = np.clip((log_spectrogram - vmin) / (vmax - vmin), 0.0, 1.0)

    # pad/crop time duration to configured width
    T = log_spectrogram.shape[1]
    if T < TARGET_W:
        log_spectrogram = np.pad(
            log_spectrogram, ((0, 0), (0, TARGET_W - T)), mode="constant"
        )
    elif T > TARGET_W:
        log_spectrogram = log_spectrogram[:, :TARGET_W]

    img = log_spectrogram.astype(np.float32)
    img = np.expand_dims(img, -1)
    return img


# ---------------------- dataset pipeline ---------------------
# load dataset and extract feature names from directory names
def load_data(df, df_type):
    features, labels = [], []
    class_example_specs = {}  # for optional visualization

    # if visualization is on, graph waveform samples
    if df_type.lower() == "training" and SHOW_VISUALS:
        get_waveform_plots(df)

    # show progress bar when loading files
    print("\n")
    progress_bar = tqdm(
        df.iterrows(),
        total=len(df),
        ncols=100,
        desc=f"Loading {df_type} Files... ",
        bar_format="{desc:<30}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
    )

    for _, row in progress_bar:
        try:
            path = get_file_path(row)  # get file path
            wav = load_file(path)  # load file from path
            img = audio_to_image(wav)  # convert wav file to spectrogram
            features.append(img)
            labels.append(row["classID"])

            # keep one spectrogram per class for visualization
            cls = row.get("class", str(row["classID"]))

            # display if visuals are on
            if SHOW_VISUALS and cls not in class_example_specs:
                class_example_specs[cls] = img.squeeze(2)  # (H, W)
        except Exception as e:
            print(f"Skipping {row.get('slice_file_name', '?')}: {e}")

    if not features:
        raise RuntimeError(
            f"No features generated for df_type={df_type} dataset. "
            f"Double check the root dataset path in 'config.py': {AUDIO_ROOT_PATH}."
        )

    X = np.stack(features) 
    y = np.array(labels)

    # if df_type.lower() == "validation" and SHOW_VISUALS and class_example_specs:
    #     plot_spectrograms(class_example_specs)

    return X, y


# ------------------------- visualization ---------------------------
# plot basic waveforms of audio for example visualization
def get_waveform_plots(df):
    class_waveforms_raw = {}
    for _, row in df.iterrows():
        label = row.get("class", str(row["classID"]))
        if label in class_waveforms_raw:
            continue
        try:
            file_path = get_file_path(row)
            raw_audio, sr = sf.read(file_path, always_2d=False)
            if raw_audio.ndim > 1:
                raw_audio = np.mean(raw_audio, axis=1)
            class_waveforms_raw[label] = (raw_audio, sr)
        except Exception as e:
            print(f"Failed to load for waveform plot: {e}")

        if len(class_waveforms_raw) == df["class"].nunique():
            break

    # if class_waveforms_raw:
    #     plot_waveform(class_waveforms_raw)
