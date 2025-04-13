import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Optional: print details about the detected GPUs
print(tf.config.list_physical_devices('GPU'))