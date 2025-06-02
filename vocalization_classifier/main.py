
# %% [markdown]
# Install all dependencies
# %%
import subprocess
from pathlib import Path
try:
    reqs = Path.home() / "CSC370" / "Assignment5.1" / "requirements.txt"
    success = subprocess.run(["pip", "install", "-r", str(reqs)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if success:
        print("Successfully installed all dependencies")
except Exception as e:
    print(f"Failed to install dependencies: {e}")

# %% [markdown]
# Setup - imports and variables
# %%
import warnings
import os
warnings.filterwarnings("ignore") 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import libraries and files
import tensorflow as tf
from termcolor import colored
from sklearn.model_selection import train_test_split
from src.ui.colors import get_acc_color, get_loss_color
from src.prep_data.get_df import build_dataframe
from src.prep_data.preprocess import load_data
from src.audio_classifier.build_model import create_classifier, train_classifier
from src.ui.cleanup import final_cleanup
# from src.checks.gpu_check import check_gpu 
# from src.prep_data.evaluate_dataset import plot_dataset
# from src.ui.visualization import visualize_stats, plot_confusion_matrix
# from src.ui.user_input import get_prediction

# force gpu usage
# assert tf.config.list_physical_devices('GPU'), "No GPU available. Exiting."

# directory paths
AUDIO_ROOT_PATH = '../dataset/combined'

# config variables
valid_split = 0.1 # % of dataset to use for validation 
BATCH_SIZE = 4 # num of files per sample 
SAMPLE_RATE = 16000 # sample rate to downsample to
DURATION_SEC = 4 # time length of audio file (seconds)
NUM_EPOCHS = 100

# %% [markdown]
# Check Device 
# %%
# check_gpu() # check if gpu is being used

# %% [markdown]
# Load and Split Dataset
# %%
df = build_dataframe(AUDIO_ROOT_PATH)
train_data, val_data = train_test_split(df, test_size=valid_split, stratify=df['classID'], random_state=42)
label_names = sorted(df['class'].unique())
num_classes = len(label_names) # get total number of classes
train_features, train_labels = load_data(AUDIO_ROOT_PATH, train_data, SAMPLE_RATE, DURATION_SEC, df_type="training")
val_features, val_labels = load_data(AUDIO_ROOT_PATH, val_data, SAMPLE_RATE, DURATION_SEC, df_type="validation")

# %% [markdown]
# Build and Train the Classifier
# %%
audio_classifier = create_classifier(num_classes)
classifier_history = train_classifier(audio_classifier, train_features, train_labels, val_features, val_labels, NUM_EPOCHS, BATCH_SIZE)
    
# %% [markdown]
# Evaluate and Plot Results
# %%
print("Final Evaluation:")
loss, acc = audio_classifier.evaluate(val_features, val_labels) # evaluate model
acc_color = get_acc_color(acc)
loss_color = get_loss_color(loss)
print(colored(f"\nValidation Loss:     {loss:.4f}\n", loss_color))
print(colored(f"Validation Accuracy: {acc:.4f} ({round((acc*100),1)}%)\n ", acc_color))

# plot training results
# plot_confusion_matrix(audio_classifier, val_features, val_labels, label_names) # create confusion matrix
# visualize_stats(classifier_history) # visualize the loss/acc

# %% [markdown]
# Get User Input and Output Prediction

# %%
# take user input, get prediction, display to user
# get_prediction(audio_classifier, SAMPLE_RATE, DURATION_SEC, label_names, user_predict_df, AUDIO_ROOT_PATH)

final_cleanup()
print("Exiting...")
