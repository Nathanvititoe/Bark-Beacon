import numpy as np
import tensorflow as tf
import os

"""
Functions for converting full model files (.keras) to tf lite models (.tflite), and then analyzing the specs of 
the tflite model to verify if it is compatible with whatever microcontroller you plan to use
"""
# convert full model to tf lite for microcontrollers
def convert_for_microcontroller(full_model_path, tflite_output_path, rep_dataset):
    model = tf.keras.models.load_model(full_model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(tflite_output_path), exist_ok=True) # Ensure directory exists
     
    # save file
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved quantized model to: {tflite_output_path}")

# define rep data for int8 quantization
def get_representative_dataset(val_features):
    def rep_data():
        for input in val_features[:100]:
            yield [np.expand_dims(input.astype(np.float32), axis=0)]
    return rep_data


