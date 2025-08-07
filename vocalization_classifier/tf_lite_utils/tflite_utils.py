import os
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
from termcolor import colored
from ai_edge_litert.interpreter import Interpreter
from src.ui.colors import get_acc_color

"""
Utility functions for testing a tf lite model, and comparing its accuracy to a full model
"""

# load the lite model for inference
def load_lite_model(tflite_path):
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

# get inference from lite model
def lite_inference(val_features, interpreter, input_details, output_details):
    tflite_preds = []

    input_dtype = input_details[0]['dtype']
    input_scale, input_zero_point = input_details[0]['quantization']

    for i in range(len(val_features)):
        float_input = val_features[i]

        if input_dtype == np.uint8 and input_scale > 0:
            # if its a quantized model
            quantized_input = (float_input / input_scale + input_zero_point).astype(np.uint8)
            input_data = np.expand_dims(quantized_input, axis=0)
        else:
            # if its a float model (no quantization)
            input_data = np.expand_dims(float_input.astype(np.float32), axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        tflite_preds.append(output_data[0])

    return np.argmax(tflite_preds, axis=1)


# function to compare full model to microcontroller lite model
def compare_models(val_features, val_labels, full_model_path, lite_model_path):
    # get full model
    model = tf.keras.models.load_model(full_model_path)
    
    interpreter, input_details, output_details = load_lite_model(lite_model_path) # load lite model
    
    # get inferences in same format
    tflite_preds_classes = lite_inference(val_features, interpreter, input_details, output_details) # get lite inference
    full_model_preds = model.predict(val_features)
    full_model_preds_classes = np.argmax(full_model_preds, axis=1)

    # get accuracy scores
    full_model_acc = accuracy_score(val_labels, full_model_preds_classes)
    tflite_acc = accuracy_score(val_labels, tflite_preds_classes)
    
    full_acc_color = get_acc_color(full_model_acc) # get full model colors
    lite_acc_color = get_acc_color(tflite_acc) # get lite model color
    
    # output results
    print("\n------------------Full Model---------------------")
    print(colored(f"Full Model Val Acc: {full_model_acc:.4f} ({round((full_model_acc*100),1)}%)\n ", full_acc_color))
    print("\n\n------------------Lite Model---------------------")
    print(colored(f"Lite Val Acc: {tflite_acc:.4f} ({round((tflite_acc*100),1)}%)\n", lite_acc_color))

    # check model size for integration
    model_size_kb = os.path.getsize(lite_model_path) / 1024
    print(f"\nQuantized TFLite model size: {model_size_kb:.2f} KB")