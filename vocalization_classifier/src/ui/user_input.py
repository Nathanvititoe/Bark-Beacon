import os
import numpy as np
from pydub import AudioSegment
import tensorflow as tf
from termcolor import colored
import subprocess

from src.prep_data.preprocess import load_file, get_yamnet_embedding
from src.ui.visualization import audio_sampler
from config import (
    AUDIO_ROOT_PATH,
    SAMPLE_RATE,
    DURATION_SEC,
    CLASS_NAMES,
)

# TODO: REWORK
"""
This is a template for prompting users for custom files the model will then make a prediction on
There are two options, upload a custom file, or randomly select one from the dataset
"""


# function to take input from user and display model prediction functionality
def get_prediction(classifier):
    getting_predictions = True
    # loop so users can get multiple predictions
    while getting_predictions:
        # prompt user
        print("\nChoose one of the following options:")
        print("1: Get a prediction on your own audio file")
        print("2: Randomly select an unseen audio sample from the test dataset")
        print("3: Exit (not esc)\n")

        inputting_choice = True
        # loop to prevent exiting without user command
        while inputting_choice:
            choice = input("Enter 1, 2 or 3: ").strip()
            is_custom_file = False  # init boolean for coloring

            # check if user wants to exit
            if choice == "3":
                print("Exiting User Interaction...")
                inputting_choice = False  # exit choice selection loop
                getting_predictions = False  # exit entirely (outer loop)
                break

            # process user file path
            elif choice == "1":
                inputting_file_path = True
                is_custom_file = True  # toggle boolean for coloring

                while inputting_file_path:
                    # get the audio file from Windows file explorer
                    filepath = get_user_file_windows()

                    # handle cancel
                    if not filepath:
                        print("User cancelled.")
                        inputting_file_path = False
                        inputting_choice = False
                        break

                    # convert non wav files
                    filepath = get_user_audio_file(filepath)
                    if filepath:
                        label = "Your File"  # label to display
                        inputting_file_path = False
                        inputting_choice = False

            # process random test file
            elif choice == "2":
                filepath, actual_label = get_random_sample(AUDIO_ROOT_PATH)
                if not filepath:
                    print("No valid samples found in dataset.")
                    continue
                label = f"Random Sample: ({actual_label})"
                is_custom_file = False  # ensure this remains false
                inputting_choice = False  # exit loop

            # handle bad inputs
            else:
                print("Invalid Input. Please enter either 1, 2 or 3 (to exit)")
                continue

        num_samples = SAMPLE_RATE * DURATION_SEC  # technical audio length

        try:
            # load the file and preprocess (returns _,1024 shape)
            waveform = load_file(filepath)
        except Exception as e:
            print(f"Failed to load or preprocess the audio file: {e}")
            return

        # display audio sample of the file
        try:
            audio_sampler(filepath, SAMPLE_RATE, DURATION_SEC, label)
        except Exception as e:
            print(f"Failed to display audio sample: {e}")

        # get model prediction
        try:
            # get yamnet embeddings
            embedding, _ = get_yamnet_embedding(waveform)
            embedding = tf.convert_to_tensor(
                embedding[None, :], dtype=tf.float32  # reshape embedding
            )
            pred = classifier.predict(embedding)  # get prediction
            pred_class = int(np.argmax(pred))  # get the class id
            confidence = pred[0][pred_class]  # get confidence level
            pred_label = CLASS_NAMES[pred_class]  # get class name

            # if low confidence, tell user (to handle user files that arent one of the 10 trained classes)
            output_confidence = round((confidence * 100), 1)
            if confidence < 0.7:
                pred_label = "unknown"
                print(
                    colored(
                        f"Warning: Low confidence prediction: {output_confidence:2f} — Defaulting to Unknown to avoid accidental corrections.",
                        "light_red",
                    )
                )
            elif confidence < 0.8:
                print(colored(f"Confidence : {output_confidence:2f}%", "yellow"))
            else:
                print(colored(f"Confidence : {output_confidence:2f}%", "green"))
            if not is_custom_file:
                pred_color = "green" if pred_label == actual_label else "red"
                print(
                    colored(
                        f"\nPrediction : {pred_label}", pred_color
                    )  # output prediction w/ color
                )
            else:
                print(f"\nPrediction : {pred_label}")  # output prediction w/o color
        except Exception as e:
            print(colored(f"Prediction failed: {e}", "red"))

        # prompt for another prediction
        response = (
            input("\nWould you like to classify another file? (y/n): ").strip().lower()
        )
        if response.strip().lower() == "n":
            getting_predictions = False


# get random file from dataset
def get_random_sample():
    files = []
    for label in CLASS_NAMES:
        label_dir = os.path.join(AUDIO_ROOT_PATH, label)
        if os.path.isdir(label_dir):
            for f in os.listdir(label_dir):
                if f.lower().endswith(".wav"):
                    files.append((os.path.join(label_dir, f), label))
    if not files:
        return None, None
    return files[np.random.randint(len(files))]


# function to convert non-wav files to wav files
def convert_files_to_wav(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)  # use AudioSegment to get the file
        audio.export(output_path, format="wav")  # export as wav file to the output path
        return output_path  # return the path the file was saved to
    except Exception as e:
        print(f"Failed to convert file '{os.path.basename(input_path)}' to WAV: {e}")
        return None


# function to retrieve the users file
def get_user_audio_file(user_input_path):
    # validate user input
    user_input_path = user_input_path.strip('"').strip("'")

    if not user_input_path or not os.path.isfile(user_input_path):
        print(f"Error: Invalid or missing file at: {user_input_path}")
        return None

    ext = os.path.splitext(user_input_path)[-1].lower()  # verify the file extension

    # if its a wav file, return the file
    if ext == ".wav":
        return user_input_path

    # if it isnt a wav file, convert it and return the converted version
    converted_name = (
        os.path.splitext(os.path.basename(user_input_path))[0]
        + ".wav"  # create name for new file
    )

    # save converted file to user test files dir (wav)
    output_dir = os.path.join(
        "dataset", "custom_test_files", "wav"  # get custom_test_files directory path
    )
    os.makedirs(output_dir, exist_ok=True)  # ensure the directory exists
    output_path = os.path.join(output_dir, converted_name)  # get path for new file

    converted = convert_files_to_wav(
        user_input_path, output_path  # convert to a wav file
    )
    return converted


def get_user_file_windows():
    ps_script = r"""
    Add-Type -AssemblyName System.Windows.Forms
    $file = New-Object System.Windows.Forms.OpenFileDialog
    $file.Filter = "Audio Files (*.wav;*.mp3;*.ogg;*.flac;*.m4a)|*.wav;*.mp3;*.ogg;*.flac;*.m4a|All Files (*.*)|*.*"
    if ($file.ShowDialog() -eq 'OK') { Write-Output $file.FileName }
    """
    result = subprocess.run(
        ["powershell.exe", "-Command", ps_script], capture_output=True, text=True
    )

    filepath = result.stdout.strip()
    if not filepath:
        print("No file selected.")
        return None

    # Convert to WSL path
    filepath = subprocess.run(
        ["wslpath", filepath], capture_output=True, text=True
    ).stdout.strip()

    if not os.path.isfile(filepath):
        print(f"Error: File not found at {filepath}")
        return None

    user_file_name = os.path.basename(filepath)
    print("-" * (len(user_file_name) + 24))
    print(f"  Loaded Custom file: {colored(user_file_name, 'green')}")
    print("-" * (len(user_file_name) + 24))
    return filepath
