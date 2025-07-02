# from tensorflow.keras import layers, models, regularizers  # type: ignore
# from tensorflow.keras.optimizers import Adam  # type: ignore
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
# from tensorflow.keras.utils import to_categorical  # type: ignore
# import tensorflow_addons as tfa
# import numpy as np
# from src.ui.cleanup import MemoryCleanupCallback


# def create_classifier_focal(num_classes):
#     model = models.Sequential([
#         layers.Input(shape=(1024,)),  # YAMNet embedding input

#         layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
#         layers.BatchNormalization(),
#         layers.Dropout(0.4),

#         layers.Dense(num_classes, activation='sigmoid')  # Sigmoid for multi-label style output
#     ])

#     model.compile(
#         optimizer=Adam(learning_rate=9e-4),
#         loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0),
#         metrics=['accuracy']
#     )

#     return model


# def train_classifier_focal(model, train_features, train_labels, val_features, val_labels, num_epochs, batch_size):
#     # One-hot encode labels for Focal Loss
#     num_classes = model.output_shape[-1]
#     train_labels = to_categorical(train_labels, num_classes=num_classes)
#     val_labels = to_categorical(val_labels, num_classes=num_classes)

#     # Callbacks
#     early_stopping = EarlyStopping(
#         monitor='val_loss',
#         patience=10,
#         min_delta=1e-8,
#         restore_best_weights=True,
#         verbose=1
#     )

#     reduce_lr = ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.5,
#         patience=3,
#         min_lr=1e-6,
#         mode='min',
#         verbose=1
#     )

#     epoch_cleanup = MemoryCleanupCallback()

#     # Train
#     history = model.fit(
#         train_features,
#         train_labels,
#         validation_data=(val_features, val_labels),
#         epochs=num_epochs,
#         batch_size=batch_size,
#         callbacks=[early_stopping, reduce_lr, epoch_cleanup],
#         verbose=2
#     )

#     return history
