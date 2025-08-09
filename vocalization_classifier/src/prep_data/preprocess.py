# external libraries
import os
import numpy as np
import scipy.signal
import soundfile as sf
from tqdm import tqdm
import librosa  # <-- new: for mel / log-mel

"""
Test file to see if we can achieve similar accuracy without using YAMNet, but still 
converting audio samples to spectrograms and then running CNN image classification on them
"""
# from src.ui.visualization import plot_spectrograms, plot_waveform
from config import SAMPLE_RATE, DURATION_SEC, AUDIO_ROOT_PATH, SHOW_VISUALS

"""
Audio -> log-mel spectrogram images (H, W, 1) for image classification (no YAMNet).
"""

# --------- spectrogram params ----------
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

# fixed time width (frames) after pad/crop so the CNN sees a static size
TARGET_W = 256
TARGET_H = N_MELS  # by definition

# ------------------------ I/O helpers ------------------------


def load_file(filepath: str) -> np.ndarray:
    """Load audio, mono, resample to SAMPLE_RATE, pad/trim to DURATION_SEC, return float32 np.array."""
    target_len = SAMPLE_RATE * DURATION_SEC
    audio, sr = sf.read(filepath, always_2d=False) # read audio sample file

    # convert all files to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # resample to 16KHz if needed
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


def _resolve_path(row) -> str:
    """Support both <root>/<class>/<file> and <root>/<file> layouts."""
    c1 = os.path.join(AUDIO_ROOT_PATH, row.get("class", ""), row["slice_file_name"])
    c2 = os.path.join(AUDIO_ROOT_PATH, row["slice_file_name"])
    if os.path.exists(c1):
        return c1
    if os.path.exists(c2):
        return c2
    raise FileNotFoundError(f"Audio file not found: {c1} | {c2}")


# ------------------ audio -> image features ------------------


def wav_to_logmel_image(waveform: np.ndarray) -> np.ndarray:
    """
    Convert 1D waveform -> log-mel image normalized to [0,1], shape (H, W, 1).
    Time axis is padded/cropped to TARGET_W frames.
    """
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
        center=False,
    )  # (n_mels, frames)

    logmel = librosa.power_to_db(mel, ref=np.max)  # (n_mels, frames)

    # robust per-file normalization to [0,1]
    vmin, vmax = np.percentile(logmel, 5), np.percentile(logmel, 95)
    if vmax <= vmin:  # degenerate edge case
        vmax = vmin + 1.0
    logmel = np.clip((logmel - vmin) / (vmax - vmin), 0.0, 1.0)

    # pad/crop time dimension to TARGET_W
    T = logmel.shape[1]
    if T < TARGET_W:
        logmel = np.pad(logmel, ((0, 0), (0, TARGET_W - T)), mode="constant")
    elif T > TARGET_W:
        logmel = logmel[:, :TARGET_W]

    img = logmel.astype(np.float32)  # (H, W)
    img = np.expand_dims(img, -1)  # (H, W, 1)
    return img


# ---------------------- dataset pipeline ---------------------


def load_data(df, df_type):
    """
    Build features/labels for a dataframe split.
    Returns: (X, y) where
      - X: np.ndarray, shape (N, TARGET_H, TARGET_W, 1)
      - y: np.ndarray, shape (N,)
    """
    features, labels = [], []
    class_example_specs = {}  # for optional plotting

    if df_type.lower() == "training" and SHOW_VISUALS:
        get_waveform_plots(df)

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
            path = _resolve_path(row)
            wav = load_file(path)  # 1D float32
            img = wav_to_logmel_image(wav)  # (H, W, 1)
            features.append(img)
            labels.append(row["classID"])

            # keep one spectrogram per class for visualization
            cls = row.get("class", str(row["classID"]))
            if SHOW_VISUALS and cls not in class_example_specs:
                # store as (H, W) for your plotting util (transpose if it expects time x freq)
                class_example_specs[cls] = img.squeeze(2)  # (H, W)
        except Exception as e:
            print(f"Skipping {row.get('slice_file_name', '?')}: {e}")

    if not features:
        raise RuntimeError(
            f"No features generated for df_type={df_type}. "
            f"Check AUDIO_ROOT_PATH='{AUDIO_ROOT_PATH}' and that files exist."
        )

    X = np.stack(features)  # (N, H, W, 1)
    y = np.array(labels)

    # if df_type.lower() == "validation" and SHOW_VISUALS and class_example_specs:
    #     # Your plot_spectrograms previously expected dict of class -> spectrogram.
    #     # If it expects (time x freq), flip/transpose as needed there.
    #     plot_spectrograms(class_example_specs)

    return X, y


# ------------------------- visuals ---------------------------


def get_waveform_plots(df):
    """Collect one raw waveform per class and plot (optional)."""
    class_waveforms_raw = {}
    for _, row in df.iterrows():
        label = row.get("class", str(row["classID"]))
        if label in class_waveforms_raw:
            continue
        try:
            file_path = _resolve_path(row)
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
