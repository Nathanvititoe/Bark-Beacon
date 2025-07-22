import numpy as np
import tensorflow as tf
import os

# convert full model to tf lite for microcontrollers
def convert_for_microcontroller(h5_model_path, tflite_output_path, rep_dataset):
    model = tf.keras.models.load_model(h5_model_path)

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


# check model size and specs to see if arduino can run it
def analyze_tflite_model(tflite_path):
    # get model file size
    model_size_kb = os.path.getsize(tflite_path) / 1024
    print(f"\n📦 TFLite Model Size: {model_size_kb:.2f} KB")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # input tensor info
    input_details = interpreter.get_input_details()
    print("\n🔍 Input Tensor(s):")
    for d in input_details:
        print(f"  - Name: {d['name']}")
        print(f"    Shape: {d['shape']}")
        print(f"    DType: {d['dtype']}")
        print(f"    Quantization: scale={d['quantization'][0]}, zero_point={d['quantization'][1]}")

    # output tensor info
    output_details = interpreter.get_output_details()
    print("\n Output Tensor(s):")
    for d in output_details:
        print(f"  - Name: {d['name']}")
        print(f"    Shape: {d['shape']}")
        print(f"    DType: {d['dtype']}")
        print(f"    Quantization: scale={d['quantization'][0]}, zero_point={d['quantization'][1]}")

    # get tensor count
    print(f"\nTotal tensors in model: {len(interpreter.get_tensor_details())}")
