from tensorflow.keras import layers, models, regularizers  # type: ignore
from tensorflow.keras.optimizers import AdamW  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from sklearn.utils import class_weight
import numpy as np
from pathlib import Path
from src.ui.cleanup import MemoryCleanupCallback
from ConvertForArduino.analyze_tflite import analyze_tflite_model
from ConvertForArduino.get_cpp_version import convert_tflite_to_cpp
from tf_lite_utils.converter.tflite_converter import convert_for_microcontroller
from config import (
    NUM_CLASSES,
    NUM_EPOCHS,
    BATCH_SIZE,
    FULL_MODEL_PATH,
    # SHOW_VISUALS,
    # USER_PREDICT,
)

# from src.ui.user_input import get_prediction
# from src.ui.visualization import plot_confusion_matrix, visualize_stats

"""
This file creates the audio classification model, trains it on the training dataset
and then saves the keras file, converts to TFLite, and converts to cpp/h files. 
Lastly, the TFLite model is ran through an analysis script to print input/output details
"""
# ---- Create CNN to classify spectrograms ----
def create_classifier(input_shape):
    if NUM_CLASSES < 1:
        raise ValueError("num_classes must be at least 1")
    if len(input_shape) != 3:
        raise ValueError(f"Expected image input (H,W,1), got {input_shape}")

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPool2D(2),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPool2D(2),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            layers.Dense(
                NUM_CLASSES,
                activation="softmax",
                kernel_regularizer=regularizers.l2(1e-4),
            ),
        ]
    )

    model.compile(
        optimizer=AdamW(learning_rate=9e-4, weight_decay=7e-2),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# train the model on the training dataset, using these configured hyperparameters
def train_classifier(
    audio_classifier, train_features, train_labels, val_features, val_labels
):
    print("\n")

    # stop early if learning slows too much
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-8,
        restore_best_weights=True,
        verbose=1,
    )

    # slow learning during plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        mode="min",
        verbose=1,
        lower_is_better=True,
    )

    # cleanup ram/gpu after each epoch
    epoch_cleanup = MemoryCleanupCallback()

    # apply weights to offset class imbalance
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels,
    )
    class_weights = dict(enumerate(weights))

    # get the model history
    history = audio_classifier.fit(
        train_features,
        train_labels,
        validation_data=(val_features, val_labels),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[epoch_cleanup, early_stopping, reduce_lr],
        verbose=2,
        class_weight=class_weights,
    )
    return history


# combined logic for creating, training and saving the model, also saves converted to tflite and cpp
def create_and_train(train_features, train_labels, val_features, val_labels):
    input_shape = tuple(train_features.shape[1:])
    audio_classifier = create_classifier(input_shape=input_shape)

    history = train_classifier(
        audio_classifier,
        train_features,
        train_labels,
        val_features,
        val_labels,
    )

    # save models
    Path(FULL_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    audio_classifier.save(FULL_MODEL_PATH)

    # export flows unchanged
    convert_for_microcontroller()
    convert_tflite_to_cpp()

    # if SHOW_VISUALS:
    #     plot_confusion_matrix(audio_classifier, val_features, val_labels)
    #     visualize_stats(history)

    analyze_tflite_model()

    # if USER_PREDICT:
    # get_prediction(audio_classifier, train_features)

    return history
