# external libraries
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import scipy.signal
import os 
import soundfile as sf
from tqdm import tqdm
from src.ui.visualization import plot_spectrograms, plot_waveform_comparison


# load YamNet as pretrained model
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
yamnet_model.trainable = False  # freeze the model (prevent training)


def load_file(filepath, sample_rate, audio_length):
    audio, sr = sf.read(filepath) # load audio files using soundfile
    
    # ensure all files are monophonic
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)  # stereo to mono

    # ensure all files use the same sample rate
    if sr != sample_rate:
        num_samples = int(sample_rate * len(audio) / sr) # technical audio length (samples per sec / # of seconds)
        audio = scipy.signal.resample(audio, num_samples)

    # Pad or trim for uniform duration
    if len(audio) < audio_length:
        pad = audio_length - len(audio) # if file duration is < audio length
        audio = np.pad(audio, (0, pad), mode='constant') # add silence to pad the file
    else:
        audio = audio[:audio_length] # trim to set duration (4sec)

    return tf.convert_to_tensor(audio, dtype=tf.float32)

# function to get embeddings and spectrograms from yamnet
def get_yamnet_embedding(waveform):
    _, embeddings, spectrogram = yamnet_model(waveform) # use yamnet to get 1024-embeddings and spectrograms
    embedding = tf.reduce_mean(embeddings, axis=0) # get mean to create feature vect
    return embedding, spectrogram
    
# get all labels and embeddings for files
def load_data_from_folds(audio_path, df, sample_rate, duration, df_type):
    target_num_samples = sample_rate * duration # technical length of file (samples per sec * # of seconds)
    yam_embeddings, labels = [], [] # init lists
    class_spectrograms = {}

    # plot waveforms for each class (raw and pre-processed)
    if df_type.lower() == "training":
        get_waveform_plots(audio_path, df, sample_rate, duration)
    
    # define tqdm progress bar for ui
    progress_bar = tqdm( df.iterrows(),
        total=len(df), # dataframe length
        ncols=100, # progress bar width
        desc=f"Loading {df_type} Files... ", # prefix
        bar_format='{desc:<30}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}' # format progress bar
    )

    # iterate through dataframe via tqdm
    for _, row in progress_bar: 
        path = os.path.join(audio_path, f"fold{row['fold']}", row['slice_file_name']) # get full file path
        try: 
            waveform = load_file(path, sample_rate, target_num_samples) # load the .wav file
            embedding, spectrogram = get_yamnet_embedding(waveform) # get the yamnet embedding/spectrogram
            label_str = row['class'] # get class name 

            # plot spectrogram for each class
            if label_str not in class_spectrograms:
                class_spectrograms[label_str] = spectrogram.numpy().T  # hold one spectrogram per class for visualization

            yam_embeddings.append(embedding.numpy()) # add the embedding mean to the list
            labels.append(row['classID']) # add the label to the list 
        # throw error if preprocessing fails
        except Exception as e: 
            print(f"Skipping {path}: {e}")

    if df_type.lower() == "validation":
        # create plot w/ a spectrogram for each class
        plot_spectrograms(class_spectrograms)

    return np.stack(yam_embeddings), np.array(labels) # return features/labels

# plot class waveforms
def get_waveform_plots(audio_path, df, sample_rate, duration):
    num_samples = sample_rate * duration # technical audio length (num samples per sec / seconds)

    # init dics to track which classes have waveforms
    class_waveforms_raw = {}
    class_waveforms_proc = {}

    # iterate through df until we have a sample for each class
    for _, row in df.iterrows():
        label = row['class'] # get label

        # add label to dict if it doesnt exist
        if label not in class_waveforms_raw:
            file_path = os.path.join(audio_path, f"fold{row['fold']}", row['slice_file_name'])

            try:
                # get raw audio 
                raw_audio, sr = sf.read(file_path) # load file
                class_waveforms_raw[label] = (raw_audio, sr) # add to dict

                # get pre processed audio
                processed_audio = load_file(file_path, sample_rate, num_samples).numpy() # load and preprocess
                class_waveforms_proc[label] = (processed_audio, sample_rate) # add to dict

            except Exception as e:
                print(f"Failed to load from {file_path}: {e}")

        # stop iterating through df once we have all classes
        if len(class_waveforms_raw) == df['class'].nunique():
            break
    
    plot_waveform_comparison(class_waveforms_raw, class_waveforms_proc) # compare raw v processed waveforms
   