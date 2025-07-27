# Bark Beacon: Dog Vocalization Classifier

A custom audio classification model for classifying dog vocalizations (bark, growl, whine, howl) using transfer learning with TensorFlow and YAMNet, and then converted to TensorFlow Lite and a C array for deployment on an Arduino Nano 33 BLE Sense Rev2.

---

##  Dataset

After retrieving the data I collected, place it in the `datasets/` folder, the dataset can be downloaded from Kaggle:

[Dog Vocalization Dataset (Kaggle)](https://www.kaggle.com/datasets/nathanvititoe/dog-vocalization-dataset)

The `/combined` directory is a collection of all the data I collected separated into the 4 distinct classes - bark, growl, whine and howl. That is what I used to train the model.  

---

## Assets

- **Images**:  
  Includes various screenshots captured throughout model development.

- **Visualizations**:  
  Generated using `matplotlib`:
  - Raw audio waveforms  
  - Mel spectrograms  
  - Class distribution of the dataset

---

## Project Structure

### `vocalization_classifier/`

#### ➤ `ConvertForArduino/`
Scripts to prepare models for Arduino microcontroller deployment:
- `get_lite_model.py`: Converts the full, trained `.h5` model to TensorFlow Lite format.
- `get_c_array.py`: Converts a `.tflite` model to a `.cc` C array (needed for Arduino inference).

#### ➤ `Models/`
Contains trained model artifacts:
- `.h5` files – Full models
- `.tflite` files – Quantized models for embedded inference
- `.cc` files – C arrays of TFLite models for Arduino integration

#### ➤ `src/`
All logic for:
- Creating the model architecture
- Training
- Data loading and preprocessing
- Dataset visualization

#### ➤ `tf_lite_utils/`
Utility functions for working with TensorFlow Lite:
- `converter.py`: For converting `.h5` to `.tflite`
- `tflite_utils.py`:
  - `load_lite_model()`: Load `.tflite` model from file
  - `lite_inference()`: Run inference with a TFLite model (for comparisons)
  - `compare_models()`: Compare `.h5` vs `.tflite` model predictions

#### ➤ `main.py`
combines the entire pipeline:
- Data preprocessing  
- Dataset splitting  
- Model training  
- Model conversion to TFLite  
- Accuracy comparison between `.h5` and `.tflite` models

---

## Utility Scripts

- **`combine_datasets.py`**  
  Combines all collected dataset folders into a single `combined` set containing only the needed classes: `bark`, `growl`, `whine`, `howl`  
  *(Class names are pulled from the filenames)*

- **`extract_audio.py`**  
  Preprocesses individual audio clips:
  - Converts all clips to mono, 16kHz  
  - Applies background noise removal using a 0.1 second sample for denoising (most samples are very short so the denoise sample must be even shorter)
