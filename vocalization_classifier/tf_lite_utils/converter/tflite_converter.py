import numpy as np
import tensorflow as tf
import os

"""
Functions for converting full model files (.keras) to tf lite models (.tflite), and then analyzing the specs of 
the tflite model to verify if it is compatible with whatever microcontroller you plan to use
"""
# convert full model to tf lite for microcontrollers (returns float model)
def convert_for_microcontroller(full_model_path, tflite_output_path):
    model = tf.keras.models.load_model(full_model_path)
    print("-------------Full Model details--------------")
    model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(tflite_output_path), exist_ok=True)
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved FLOAT model to: {tflite_output_path}")

# define rep data for int8 quantization
def get_representative_dataset(val_features):
    def rep_data():
        for input in val_features[:100]:
            yield [np.expand_dims(input.astype(np.float32), axis=0)]
    return rep_data


