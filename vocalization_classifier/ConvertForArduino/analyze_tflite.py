import os
import argparse
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import math

"""
can be run as a standalone script or a callable function to analyze a saved tf lite model (.tflite) 
and output its size, input tensors, output tensors and estimated tensor arena size
"""

# get a rough estimate of the tensor arena size the model will require
def estimate_tensor_arena(interpreter):
    tensor_details = interpreter.get_tensor_details()
    total_tensor_bytes = 0
    for t in tensor_details:
        shape = t['shape']
        dtype = t['dtype']
        if shape is None or len(shape) == 0:
            continue
        num_elements = np.prod(shape)
        bytes_per_elem = np.dtype(dtype).itemsize
        tensor_size = num_elements * bytes_per_elem
        total_tensor_bytes += tensor_size
    print(f"\nTotal Tensor bytes: {total_tensor_bytes:.2f}\n")
    estimated_arena = int(total_tensor_bytes * 4 + 4096)
    return estimated_arena


# check model specs to see if arduino can run it
# outputs model size, input tensor details, output tensor details, and estimated/recommended tensor arena size 
def analyze_tflite_model(tflite_path):
    model_size_kb = os.path.getsize(tflite_path) / 1024

    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    estimated_arena_size = estimate_tensor_arena(interpreter)
    print("\n-----------------------")
    print(estimated_arena_size)
    print("-----------------------\n") 
     
    # round up and give 50% buffer
    recommended_arena_kb = math.ceil(estimated_arena_size / 1024 * 1.5)
    recommended_arena_bytes = recommended_arena_kb * 1024
    
    print(f"\n\nModel Size: {model_size_kb:.2f} KB\n")

    print("Input Tensor:")
    for d in input_details:
        input_len = np.prod(d['shape'])
        print(f"  Shape: {d['shape']}")
        print(f"  Type: {d['dtype']}")
        print(f"  Input Length (flattened): {input_len}")
        print(f"  Quantization: scale={d['quantization'][0]}, zero_point={d['quantization'][1]}")

    print("\nOutput Tensor:")
    for d in output_details:
        print(f"  Shape: {d['shape']}")
        print(f"  Type: {d['dtype']}")
        print(f"  Quantization: scale={d['quantization'][0]}, zero_point={d['quantization'][1]}")
    
    print(f"\nEstimated Tensor Arena: {recommended_arena_bytes / 1024:.2f} KB")
    print(f"Recommended Arena (+50% buffer): {recommended_arena_kb} KB\n")
    print(f"Use: `constexpr int tensorArenaSize = {recommended_arena_bytes};` in Arduino code\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a TFLite model for Arduino deployment")
    parser.add_argument("model_path", type=str, help="Path to the .tflite model file")
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        print(f"File not found: {args.model_path}")
        exit(1)

    analyze_tflite_model(args.model_path)