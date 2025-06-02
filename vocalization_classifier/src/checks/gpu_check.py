# function to check if gpu is being used
# def check_gpu():
#     gpus = tf.config.list_physical_devices('GPU') # list devices tf sees
#     if gpus:
#         details = tf.config.experimental.get_device_details(gpus[0])
#         gpu_name = details.get("device_name", "GPU:0")  # use name or GPU:0 
#         print(f"\n\nTensorFlow is using device: {gpu_name}\n")
#     else:
#         print("\n\nTensorFlow is NOT using a GPU.\n")