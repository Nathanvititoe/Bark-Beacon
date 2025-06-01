from tensorflow.keras import layers, models, regularizers # type: ignore
from tensorflow.keras.optimizers import AdamW, Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau # type: ignore
from src.ui.cleanup import MemoryCleanupCallback

# create the simple CNN to classify yamnet embeddings
def create_classifier(num_classes):
    # build classifier w/ input shape same as yamnet output shape
    audio_classifier = models.Sequential([
        # custom classifier head
        layers.Input(shape=(1024,)),  # input yamnet 1024-embedding (feature vector)

        # conv layer
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-3)), # dense layer, 256 neurons/nodes
        layers.BatchNormalization(), # normalize each batch
        layers.Dropout(0.4), # randomly drop 40% of neurons (prevents overfitting)
        
        # output layer
        layers.Dense(num_classes, activation='softmax')  # dense output layer, output num_classes
    ])
    
    # compile the model
    audio_classifier.compile(
        optimizer=AdamW(learning_rate=9e-4, weight_decay=7e-2), # use adam for optimization
        loss='sparse_categorical_crossentropy', # categorical crossentropy for multi-classification
        metrics=['accuracy'] # measure accuracy
    )

    return audio_classifier

# function to train and validate the classifier 
def train_classifier(audio_classifier, train_features, train_labels, val_features, val_labels, num_epochs, batch_size):
    # define callbacks to refine training
    early_stopping = EarlyStopping(
        monitor='val_loss', # what to monitor
        patience=10, # wait this many epochs without improvement
        min_delta=1e-8, # min gain to achieve w/o stopping
        restore_best_weights=True,  # use best epoch after stopping
        verbose=1 # output logs
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', # what metric to monitor
        factor=0.5, # reduce LR by this factor
        patience=3, # wait this many epochs 
        min_lr=1e-6, # stop reducing at this LR
        mode='min', # try to achieve lower loss
        verbose=1, # turn on logs
        lower_is_better=True # try to achieve lowest val_loss
    )
    # add custom callbacks
    epoch_cleanup = MemoryCleanupCallback()

    # train the model
    classifier_history = audio_classifier.fit(
        train_features, # data to train on
        train_labels,  # labels
        validation_data=(val_features, val_labels), # what to validate with
        epochs=num_epochs, # how many epochs to train
        batch_size=batch_size, # size of batches
        callbacks=[ # define callbacks (called every epoch)
            epoch_cleanup,  # custom cleanup for resource optimization
            early_stopping,  # stops when training becomes stagnant
            reduce_lr # reduces learning rate as training slows
            ], 
        verbose=2 # turn on training logs
        )
    
    return classifier_history