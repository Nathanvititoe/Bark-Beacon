from tensorflow.keras import layers, models, regularizers  # type: ignore
from tensorflow.keras.optimizers import AdamW  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from sklearn.utils import class_weight
import numpy as np
from src.ui.cleanup import MemoryCleanupCallback
from ConvertForArduino.analyze_tflite import analyze_tflite_model
from ConvertForArduino.get_cpp_version import (
    convert_tflite_to_cpp,
)
from tf_lite_utils.converter.tflite_converter import (
    convert_for_microcontroller,
)
from config import (
    NUM_CLASSES,
    NUM_EPOCHS,
    BATCH_SIZE,
    FULL_MODEL_PATH,
    LITE_MODEL_PATH,
    SHOW_VISUALS,
    LABEL_NAMES,
    USER_PREDICT,
)
from src.ui.user_input import get_prediction
from src.ui.visualization import (
    plot_confusion_matrix,
    visualize_stats,
)

"""
This file has callable functions for creating the audio classification model, as well as training it,
and allowing validation testing by returning the model_history from the training function
"""


# create the simple CNN to classify yamnet embeddings
def create_classifier():
    if NUM_CLASSES < 1:
        raise ValueError("num_classes must be at least 1")

    # build classifier w/ input shape same as yamnet output shape
    audio_classifier = models.Sequential(
        [
            # custom classifier head
            layers.Input(shape=(1024,)),  # input yamnet 1024-embedding (feature vector)
            # used 256 for best accuracy
            layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=regularizers.l2(
                    1e-3
                ),  # dense layer, 256 neurons/nodes
            ),
            # dropped to 128 to reduce model size for embedded use (still ~94% accuracy)
            # layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-3)), # dense layer, 256 neurons/nodes
            layers.BatchNormalization(),  # normalize each batch
            layers.Dropout(0.4),  # randomly drop 40% of neurons (prevents overfitting)
            # output layer
            layers.Dense(
                NUM_CLASSES, activation="softmax"
            ),  # dense output layer w/ softmax for categorical cross
        ]
    )

    # compile the model
    audio_classifier.compile(
        optimizer=AdamW(
            learning_rate=9e-4, weight_decay=7e-2  # use adamW for optimization
        ),
        loss="sparse_categorical_crossentropy",  # categorical crossentropy for multi-classification
        metrics=["accuracy"],  # measure accuracy
    )

    return audio_classifier


# function to train and validate the classifier
def train_classifier(
    audio_classifier, train_features, train_labels, val_features, val_labels
):
    print("\n")  # start training w new line

    # define callbacks to refine training
    early_stopping = EarlyStopping(
        monitor="val_loss",  # what to monitor
        patience=10,  # wait this many epochs without improvement
        min_delta=1e-8,  # min gain to achieve w/o stopping
        restore_best_weights=True,  # use best epoch after stopping
        verbose=1,  # output logs
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",  # what metric to monitor
        factor=0.5,  # reduce LR by this factor
        patience=3,  # wait this many epochs
        min_lr=1e-6,  # stop reducing at this LR
        mode="min",  # try to achieve lower loss
        verbose=1,  # turn on logs
        lower_is_better=True,  # try to achieve lowest val_loss
    )
    # add custom callbacks
    epoch_cleanup = MemoryCleanupCallback()

    # get class weights to balance out the extensive bark samples
    weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(train_labels), y=train_labels
    )
    class_weights = dict(enumerate(weights))
    # train the model
    classifier_history = audio_classifier.fit(
        train_features,  # data to train on
        train_labels,  # labels
        validation_data=(val_features, val_labels),  # what to validate with
        epochs=NUM_EPOCHS,  # how many epochs to train
        batch_size=BATCH_SIZE,  # size of batches
        callbacks=[  # define callbacks (called every epoch)
            epoch_cleanup,  # custom cleanup for resource optimization
            early_stopping,  # stops when training becomes stagnant
            reduce_lr,  # reduces learning rate as training slows
        ],
        verbose=2,  # turn on training logs
        class_weight=class_weights,
    )

    return classifier_history


"""
unified function to build and train the full classifier and then save the full model,
convert to tflite model and then finally convert to a C array for Arduino
returns the full model history
"""


def create_and_train(train_features, train_labels, val_features, val_labels):
    audio_classifier = create_classifier(NUM_CLASSES)
    classifier_history = train_classifier(
        audio_classifier,
        train_features,
        train_labels,
        val_features,
        val_labels,
        NUM_EPOCHS,
        BATCH_SIZE,
    )
    # save models
    audio_classifier.save(FULL_MODEL_PATH)  # save full model to .keras file
    convert_for_microcontroller(FULL_MODEL_PATH, LITE_MODEL_PATH)  # get tflite model
    convert_tflite_to_cpp(LITE_MODEL_PATH)  # get cpp files for tflite model

    # plot training results
    if SHOW_VISUALS:
        plot_confusion_matrix(
            audio_classifier,
            val_features,
            val_labels,
            LABEL_NAMES,  # create confusion matrix
        )
        visualize_stats(classifier_history)  # visualize the loss/acc

    # print tflite model specs for using in Arduino
    analyze_tflite_model(LITE_MODEL_PATH)

    if USER_PREDICT:
        # prompt user to input files for model prediction
        get_prediction(audio_classifier, train_features)
    return classifier_history
