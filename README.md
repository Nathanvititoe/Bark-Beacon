# Bark Beacon: Dog Vocalization Classifier

A custom audio classification model for classifying dog vocalizations (bark, growl, whine, howl) using with TensorFlow 

After saving the full model in a .keras file, it is converted to TensorFlow Lite and a C++/header file so that it can be used on development boards like the Arduino Nano 33 BLE Sense Rev2 or ESP32.

---

##  Dataset

After retrieving the data I collected, place it in the `datasets/` folder, the dataset can be downloaded from Kaggle:

[Dog Vocalization Dataset (Kaggle)](https://www.kaggle.com/datasets/nathanvititoe/dog-vocalization-dataset)

The `/combined` directory is a collection of all the data I collected separated into the 4 distinct classes - bark, growl, whine and howl. That is what I used to train the model.  

## Tensorflow
To set up Tensorflow on 50 series GPU's follow this guide to run it all in Docker
[Tensorflow 50 series Guide](https://blog.mypapit.net/2025/06/how-to-get-tensorflow-acceleration-with-nvidia-rtx-50-series-gpu-with-docker-rtx5060ti-16gb-for-ubuntu-and-windows-wsl2.html)

---
### THIS IS OUT OF DATE FOR THE CURRENT FILE STRUCTURE
#### ➤ `images/`:  
  Includes various screenshots captured throughout model development.

#### ➤ `Models/`
Contains trained model artifacts:
- `.keras` files – Full models
- `.tflite` files – Quantized models for embedded inference
- `.cpp` files – C++ arrays of TFLite models for Arduino integration
- `.h` files - header files that pair with the C++ models
On GitHub this only contains the `latest` directory, converted .cpp/.h files from the latest TFLite model
However, in development, everytime main is run, this folder is populated with a .keras file (full model), .tflite file (lite model), and a directory containing a .cpp and .h for the latest version created (labeled as "_v(x)")
  
#### ➤ `old_yam_build/`
  this directory is not used anywhere else in the application, it is simply saved versions of the preprocessing and model build from when I utilized YAMNet for transfer learning, in case I ever need to utilize it. 

#### ➤ `standalone_scripts/`
  Contains scripts used to assist in data collection, for increasing the size of the dataset, therefore increasing model accuracy. 
  
  - *combine_datasets.py* : walks through all directories in the dataset directory and combines their files into a single `combined` folder, creating subdirectories for each class (bark, growl, howl, whine) and inferring which files belong to each class based on their file names. 
  - *extract_audio.py* : used to convert a .mp4 video file to a .wav audio format so that it can be used as an audio sample within the dataset
  - *urbanaudiopull.py* : quick, simple script to pull the UrbanSound8k dataset from kaggle using an api key, it then extracts 600 files, evenly distributed between UrbanSound8k classes (defined in the csv), excluding any classes related to dogs, it then places them in `dataset/combined` under the `unknown` subdirectory as a collection of unknown samples for the model to differentiate dog vocalizations from other sounds. 

#### ➤ `visualizations/`
  Generated using `matplotlib`:
  - Raw audio waveforms  
  - Mel spectrograms  
  - Class distribution of the dataset

---

## Project Structure

### `vocalization_classifier/`

#### ➤ `arduino_conversion/`
Scripts to prepare models for Arduino microcontroller deployment:
- `get_lite_model.py`: Converts the full, trained `.keras` model to TensorFlow Lite format.
- `get_cpp_version.py`: Converts a `.tflite` model to a `.cpp` array (needed for Arduino inference) with a matching header file (`.h`).
- `analyze_tflite.py`: Analyzes the saved `.tflite` file from within the `/models` directory, outputting input/output tensor details, as well as a rough estimate of the required tensor arena size, to assist in Arduino IDE configuration (embedded code)

#### ➤ `src/`
All logic for:
- Data loading and preprocessing
  - `/prep_data` - contains all files and logic used for loading audio files, making them uniform, splitting into training and validation sets, and converting to a dataframe
- Creating/Training the Audio Classifier
  - `/audio_classifier` - all files and logic for creating and training the model
- Dataset visualization
  - `/ui` - collection of functions/files for displaying data in matplotlib graphs and matrices
- 
#### ➤ `test_audio/`
Small collection of audio files not found within the dataset, used for automated and manual testing
Unknown samples are used to test the model on audio that isnt a dog vocalization, verifying that it infers "unknown"
- dog barks (2)
- dog growl (1) 
- dog howl (1)
- dog whine (1)
- wolf howl (1) : (for confidence test)
- baby crying (1) : unknown sample
- dishes clanking (1) : unknown sample
- music (1) : unknown sample
- people talking (1) : unknown sample

#### ➤ `tests/`
  - `/unit` : collection of unit tests for verifying the functionality and outputs of the preprocessing pipeline
  - `/integration` : tests the data pipeline end to end
  - `/performance` : simple tests for checking model size/footprint and latency are under configured maximum values (the best test for model performance is validation accuracy)
  - `configure_tests.py` : sets testing variables and configuration

#### ➤ `tf_lite_utils/`
Utility functions for working with TensorFlow Lite:
- `converter.py`: For converting `.keras` to `.tflite`
- `tflite_utils.py`:
  - `load_lite_model()`: Load `.tflite` model from file
  - `lite_inference()`: Run inference with a TFLite model (for comparisons)
  - `compare_models()`: Compare `.keras` vs `.tflite` model predictions

#### ➤ `main.py`
combines the entire pipeline:
- Data preprocessing  
- Dataset splitting  
- Model training  
- Model conversion to TFLite  
---


# Running on Arduino or ESP32 in the Arduino IDE
**Reference:** [Arduino Nano 33 BLE Sense ML Guide](https://docs.arduino.cc/tutorials/nano-33-ble-sense/get-started-with-machine-learning/)

### If the model is already trained
1. Copy the appropriate `.cpp` and `.h` files from:  
   `vocalization_classifier/models/`  
   to your Arduino sketch folder.
2.  In your Arduino sketch:
   - Include the copied files using `#include`.
   - Use the **recommended Arduino libraries** for running inference (e.g., `Arduino_TensorFlowLite`).

### If you're starting from scratch
1. Place the dataset in the `dataset/` directory.
2. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```
- Run the training and conversion pipeline:

   ```bash
   python vocalization_classifier/main.py
   ```
    This will:
   - Preprocess the dataset
   - Train the full model
   - Generate:
     - `Bark_Beacon_Full.keras`
     - `Bark_Beacon_Lite.tflite`
     - `audio_classifier_v(?).cpp` and `audio_classifier_v(?).h` for Arduino
- Copy the generated `.cpp` and `.h` files from:

   ```
   vocalization_classifier/models/{your_version}
   ```
   or
   ```
   vocalization_classifier/models/latest
   ```
   to your Arduino sketch folder.
- In your Arduino sketch, include the model files and set up inference as described in the Arduino ML guide:

   ```cpp
   #include "audio_classifier.h"
   ```
