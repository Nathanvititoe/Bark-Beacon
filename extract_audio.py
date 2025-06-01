import os
from pydub import AudioSegment
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np

mp4_root = "dataset/mp4_25May2025" # mp4 dir
wav_root = "dataset/wav_25May2025" # new wav dir

# loop through mp4 directory and subdirectories
for root, _, files in os.walk(mp4_root):
    for file in files:
        if not file.lower().endswith(".mp4"):
            continue

        mp4_path = os.path.join(root, file)

        # create copy dir
        # Create mirrored output path with wav name
        relative_path = os.path.relpath(root, mp4_root)
        output_dir = os.path.join(wav_root, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        wav_filename = os.path.splitext(file)[0] + ".wav"
        wav_path = os.path.join(output_dir, wav_filename)

        print(f"Processing: {mp4_path} → {wav_path}")

        try:
            # extract audio, convert to wav
            audio = AudioSegment.from_file(mp4_path, format="mp4")

             # Force mono, 16kHz, 16-bit
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

            temp_wav_path = wav_path + ".temp.wav"
            audio.export(temp_wav_path, format="wav")

            # Load WAV and remove bg noise
            y, sr = librosa.load(temp_wav_path, sr=None)
            # Check if clip is long enough for a noise sample
            if len(y) < int(sr*0.2):
                print(f"Skipping noise reduction for {wav_path} — too short.")
                continue
            noise_sample = y[:int(sr * 0.1)]  # first 0.1s for noise sample
            cleaned = nr.reduce_noise(
                y=y,
                sr=sr,
                y_noise=noise_sample,
                prop_decrease=0.5,
                stationary=True
            )

            # Save denoised audio
            sf.write(wav_path, cleaned, sr)
            os.remove(temp_wav_path)  # Clean up temp

        except Exception as e:
            print(f"Failed to process {mp4_path}: {e}")
