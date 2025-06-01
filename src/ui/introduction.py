from tqdm import tqdm
from termcolor import colored

# description of the model for user (using tqdm for color/ui)
def introduction():
    tqdm.write(colored("\nUrbanAudio8k Audio Classifier", "red", attrs=["bold"]))
    tqdm.write(colored("--------------------------------\n", "red"))

    tqdm.write(colored(
        "This is an audio classifier built using TensorFlow and YAMNet, it's trained on the Kaggle UrbanSound8K dataset to detect different environmental sounds like sirens,\n"
        "car horns, dog barks, etc.\n", "green", attrs=["bold"]))

    tqdm.write(colored("Dataset :\n", "yellow"))
    tqdm.write(colored("   UrbanSound8K: ", "green", attrs=["bold"]) + colored("https://www.kaggle.com/datasets/chrisfilo/urbansound8k", "blue", attrs=["underline"]))

    tqdm.write(colored("\n How it works :\n", "yellow"))

    steps = [
        "1. We load raw audio waveforms from the dataset",
        "2. Preprocess those files for uniformity, make them all mono, 16kHz and 4 seconds",
        "3. Convert those waveforms into log-mel spectrograms using Google's YAMNet",
        "4. YAMNet uses a pretrained MobileNet CNN to get dimensional embeddings with 1024 feature vectors",
        "5. We feed the YAMNet output (1024-embeddings) into our custom dense classifier layer",
        "6. Our classifier layer will output a prediction (e.g., 'siren', 'drilling', 'dog_bark')"
    ]
    for step in steps:
        tqdm.write(colored(f"   {step}", "green", attrs=["bold"]))

# returns a color based on the accuracy number (target >= 92%)
def get_acc_color(acc):
    if acc >= 0.92:
        return "green"
    elif acc >= 0.85:
        return "yellow"
    else:
        return "red"

# returns a color based on the loss number (target <= 0.5)    
def get_loss_color(loss):
    if loss <= 0.5:
        return "green"
    elif loss <= 0.75:
        return "yellow"
    else:
        return "red"