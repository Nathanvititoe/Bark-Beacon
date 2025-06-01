import os
import numpy as np
from pydub import AudioSegment
import tensorflow as tf
from termcolor import colored
from src.prep_data.preprocess import load_file, get_yamnet_embedding
from src.ui.visualization import audio_sampler

# tested custom files pulled from internet (not included in dataset)
#   gun shots                      ✔
#   car-horn                       ✔
#   engine idle                    ✔
#   old-car horn                   ✔
#   fire-siren                     ✔
#   police-sirens                  ✔
#   untrained_sound (baby coo)     ✔
#   untrained_sound2 (wind chimes) ✔


# function to take input from user and display model prediction functionality
def get_prediction(classifier, sample_rate, duration, class_names, user_predict_df, audio_root_path):
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
            is_custom_file = False # init boolean for coloring
            # check if user wants to exit 
            if choice == "3":
                print("Exiting User Interaction...")
                inputting_choice = False # exit choice selection loop
                getting_predictions = False # exit entirely (outer loop)
                break
            
            # process user file path
            elif choice == "1":
                inputting_file_path = True 
                is_custom_file = True # toggle boolean for coloring
                # loop to get file path / prevent exiting
                while inputting_file_path:
                    # get user filepath as input
                    filepath = input("Please enter the file path for your .wav file, relative to the CSC370 directory (or 'exit' to cancel): ").strip()
                    
                    # check if user wants to exit
                    if filepath.lower() == "exit":
                        print("User cancelled.")
                        inputting_file_path = False # exit file path input loop
                        inputting_choice = False # exit choice selection loop
                    
                    # get the audio file from the user provided path
                    filepath = get_user_audio_file(filepath)

                    # if its a good file, exit both loops
                    if filepath:
                        label = "Your File" # label to display
                        inputting_file_path = False # exit file path input loop
                        inputting_choice = False # exit choice selection loop

                
            # process unseen test file from user_predict_df
            elif choice == "2":
                row = user_predict_df.sample(n=1).iloc[0] # get a random sample from user_predict_df
                filepath = os.path.join(audio_root_path, f"fold{row['fold']}", row['slice_file_name']) # get the full filepath
                label = f"Unseen Sample: ({row['class']})" # get label to display
                actual_label = row['class']
                is_custom_file = False # ensure this remains false
                inputting_choice = False # exit loop

            # handle bad inputs
            else:
                print("Invalid Input. Please enter either 1, 2 or 3 (to exit)")   
                continue
        
        num_samples = sample_rate * duration # technical audio length

        try:
            # load the file and preprocess (returns _,1024 shape)
            waveform = load_file(filepath, sample_rate, num_samples)
        except Exception as e:
            print(f"Failed to load or preprocess the audio file: {e}")
            return

        # display audio sample of the file
        try:
            audio_sampler(filepath, sample_rate, duration, label)
        except Exception as e:
            print(f"Failed to display audio sample: {e}")

        # get model prediction
        try:
            # get yamnet embeddings 
            embedding, _ = get_yamnet_embedding(waveform)
            embedding = tf.convert_to_tensor(embedding[None, :], dtype=tf.float32)  # reshape embedding
            pred = classifier.predict(embedding) # get prediction
            pred_class = int(np.argmax(pred)) # get the class id
            confidence = pred[0][pred_class] # get confidence level
            pred_label = class_names[pred_class] # get class name
            if is_custom_file == False:
                pred_color = "green" if pred_label == actual_label else "red"
                print(colored(f"\nPrediction : {pred_label}", pred_color)) # output prediction w/ color
            else:
                print(f"\nPrediction : {pred_label}") # output prediction w/o color

            
            # if low confidence, tell user (to handle user files that arent one of the 10 trained classes)
            output_confidence = round((confidence * 100), 1)
            if confidence < 0.7:
                print(colored(f"Warning: Low confidence prediction: {output_confidence} — this audio may not match any known class.", "light_red"))
            elif confidence < 0.8:
                print(colored(f"Confidence : {output_confidence}%", "yellow")) 
            else:
                print(colored(f"Confidence : {output_confidence}%", "green")) 
        except Exception as e:
            print(colored(f"Prediction failed: {e}", "red"))
        
        # prompt for another prediction
        response = input("\nWould you like to classify another file? (y/n): ").strip().lower()
        if response.strip().lower() == "n":
            getting_predictions = False

# function to convert non-wav files to wav files
def convert_files_to_wav(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path) # use AudioSegment to get the file
        audio.export(output_path, format="wav") # export as wav file to the output path
        return output_path # return the path the file was saved to
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
    else:
        # output to user that load was successful
        user_file_name = os.path.basename(user_input_path)
        path_length = len(user_file_name)
        print("-" * (path_length + 24))
        print(f"  Loaded Custom file: {colored(user_file_name, 'green')}")
        print("-" * (path_length + 24))

    ext = os.path.splitext(user_input_path)[-1].lower() # verify the file extension
    
    # if its a wav file, return the file
    if ext == ".wav": 
        return user_input_path

    # if it isnt a wav file, convert it and return the converted version
    converted_name = os.path.splitext(os.path.basename(user_input_path))[0] + ".wav" # create name for new file
    
    # save converted file to user test files dir (wav)
    output_dir = os.path.join("dataset", "custom_test_files", "wav") # get custom_test_files directory path
    os.makedirs(output_dir, exist_ok=True)  # ensure the directory exists
    output_path = os.path.join(output_dir, converted_name) # get path for new file

    converted = convert_files_to_wav(user_input_path, output_path) # convert to a wav file
    return converted
