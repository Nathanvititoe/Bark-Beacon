from sklearn.model_selection import train_test_split
from vocalization_classifier.src.prep_data.preprocess import load_data
from config import VALID_SPLIT

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
    train_features, train_labels = load_data(train_data, df_type="training")

    # preprocess/load the validation data
    val_features, val_labels = load_data(val_data, df_type="validation")

    return train_features, train_labels, val_features, val_labels
