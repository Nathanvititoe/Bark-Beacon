from sklearn.model_selection import train_test_split
from src.prep_data.preprocess import load_data
from imblearn.over_sampling import RandomOverSampler
from config import AUDIO_ROOT_PATH, VALID_SPLIT, SAMPLE_RATE, DURATION_SEC

"""
This will split the dataset into training and validation sets, and run each df through its own preprocessing,
returning the training features/labels and the validation features/labels
 """


def get_train_val_split(df):
    # split dataset
    train_data, val_data = train_test_split(
        df, test_size=VALID_SPLIT, stratify=df["classID"], random_state=42
    )

    # preprocess/load the training data
    train_features, train_labels = load_data(
        AUDIO_ROOT_PATH, train_data, SAMPLE_RATE, DURATION_SEC, df_type="training"
    )

    # preprocess/load the validation data
    val_features, val_labels = load_data(
        AUDIO_ROOT_PATH, val_data, SAMPLE_RATE, DURATION_SEC, df_type="validation"
    )

    # oversample to help with class imbalance
    ros = RandomOverSampler(random_state=42)
    train_features, train_labels = ros.fit_resample(train_features, train_labels)

    return train_features, train_labels, val_features, val_labels
