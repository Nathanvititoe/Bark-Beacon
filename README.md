# Video Link: 
https://youtu.be/icp3g757wzU

# Instructions: 
    This project must be run via Jupyter notebook or another GUI due to the extensive plotting and visualization

## Dataset : 
https://www.kaggle.com/datasets/chrisfilo/urbansound8k

## How to Run the Notebook:
    - Create python venv in the CSC370/Assignment5.1 directory
    - Navigate to "main.ipynb"
    - Select your created venv as the python kernel
    - Select "Run All Cells"

## How it Works: 
    YAMNet:
        - takes in raw audio waveforms and converts them into a log-mel spectrogram (visualization of how frequency changes over time).
        
        - uses that spectrogram and feeds it through a Convolutional Neural Network (CNN) that was pre-trained on "Audioset" (large dataset)
        
        - outputs a set of feature vectors with input shape (1024)

    Custom:
        - Remove the YAMNet classifier (top layer)

        - Create a custom classifier (top layer) and use the YAMNet output as input (input_shape=1024)

            -  Single Convolutional Layer 
                    -Dense Layer (256 neurons)
                    -ReLu activation function
            
            - Output Layer with output_shape as the number of classes we are trying to classify (using softmax function)
                 
### How I Would Continue (given time):
    - Focus on reducing the mild overfitting towards the end of training
        - Test different regularization (L2, dropout levels)
        - Add more audio augmentation (time shifting, pitch shifts)
        - Test different classifier head depths

## Test Files ('/dataset/custom_test_files/{file_type})
#### WAV Files (accepted format)
    - car-horn.wav
    - engine-idle.wav
    - fire-siren.wav
    - old-car-horn.wav
    - pistol-shots.wav
    - police-sirens.wav
    
    Included Baby Noise to test the output for audio the model wasn't trained on:
        - untrained_sound.wav 

#### MP3 Files (for file conversion testing)
    - car-horn.mp3
    - engine-idle.mp3
    - fire-siren.mp3
    - old-car-horn.mp3
    - pistol-shots.mp3
    - police-sirens.mp3


## Resources :

##### YAMNet & Tensorflow specific -- Audio Classification
- [Audio Classification using Google’s YAMNet – GeeksForGeeks](https://www.geeksforgeeks.org/audio-classification-using-googles-yamnet/)
- [Transfer Learning for Audio – TensorFlow Tutorial](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio)
- [Intro to Audio Classification with TensorFlow – Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/intro-audio-classification-tensorflow/)

##### Examples of Same Dataset
- [Urban Sound Classification – Satoru90 (Kaggle)](https://www.kaggle.com/code/satoru90/urban-sound-classification)
- [Urban8K TensorFlow Audio Classifier – Pulkit Dhingra (Kaggle)](https://www.kaggle.com/code/pulkit12dhingra/urban-8k-tensorflow-audio-classification)

##### General Audio Classification
- [How to Classify Sound Using FFT and Neural Networks (StackOverflow)](https://stackoverflow.com/questions/55977956/how-to-classify-sound-using-fft-and-neural-network-should-i-use-cnn-or-rnn)
- [Classify MNIST Audio using Spectrograms + Keras CNN – Christian Lillelund (Kaggle)](https://www.kaggle.com/code/christianlillelund/classify-mnist-audio-using-spectrograms-keras-cnn)

##### For UI and Visualization
- [TQDM – Progress Bars for Python](https://tqdm.github.io/)
- [Python termcolor](https://pypi.org/project/termcolor/)
- [Matplotlib API Docs](https://matplotlib.org/stable/api/index.html)
- [IPython API Docs](https://ipython.org/documentation.html)
- [SciKit-learn: Confusion Matrix Display](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html)