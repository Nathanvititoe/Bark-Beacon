import sys
from unittest.mock import MagicMock
sys.modules['src.ui.cleanup'] = MagicMock()

import pytest
import numpy as np
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.callbacks import History # type: ignore
from vocalization_classifier.src.audio_classifier.build_model import create_classifier, train_classifier


@pytest.fixture
def dummy_data():
    # create small mock dataset for testing
    train_features = np.random.rand(10, 1024).astype(np.float32)
    train_labels = np.random.randint(0, 2, size=(10,))
    val_features = np.random.rand(4, 1024).astype(np.float32)
    val_labels = np.random.randint(0, 2, size=(4,))
    return train_features, train_labels, val_features, val_labels

#  Test model creation, structure and hyperparams
def test_create_classifier_structure():
    num_classes = 2
    model = create_classifier(num_classes) # create mock model
    assert isinstance(model, Model) # verify creation
    assert model.input_shape == (None, 1024) # verify input shape
    assert model.output_shape[-1] == num_classes # verify output shape
    assert model.optimizer.__class__.__name__ == "AdamW" # verify optimizer
    assert model.loss == "sparse_categorical_crossentropy" # verify loss function

# verify the model can train without errors, and return its history
def test_train_classifier_returns_history(dummy_data):
    train_features, train_labels, val_features, val_labels = dummy_data 
    model = create_classifier(num_classes=2) # create mock model
    history = train_classifier(model, train_features, train_labels, val_features, val_labels, num_epochs=1, batch_size=2)
    assert isinstance(history, History) 
    assert "loss" in history.history
    assert "val_loss" in history.history

# test prediction output shape
def test_model_prediction_shape(dummy_data):
    train_features, _, _, _ = dummy_data
    model = create_classifier(num_classes=2)
    predictions = model.predict(train_features, verbose=0)
    assert predictions.shape == (10, 2)
    assert np.all(predictions >= 0) and np.all(predictions <= 1)  # softmax range

# edge case test for an invalid number of classes
def test_create_classifier_invalid_classes():
    with pytest.raises(ValueError):
        create_classifier(0)  # for success, must be >= 1

# test a known problem, too small of a dataset cant be split
@pytest.mark.xfail(reason="Training on too small of a dataset will fail because there isnt enough data to split training and validation.")
def test_train_classifier_single_sample():
    train_features = np.random.rand(1, 1024).astype(np.float32)
    train_labels = np.array([0])
    val_features = np.random.rand(1, 1024).astype(np.float32)
    val_labels = np.array([0])
    model = create_classifier(num_classes=2)
    history = train_classifier(model, train_features, train_labels, val_features, val_labels, num_epochs=1, batch_size=1)
    assert "loss" in history.history
