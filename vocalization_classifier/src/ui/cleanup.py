import gc
import tensorflow as tf
from tensorflow.keras.callbacks import Callback # type: ignore
from numba import cuda

# cleanup to run at the end of model use
def final_cleanup():
    print("\n\nCleaning up...\n")

    # clear tf session 
    tf.keras.backend.clear_session()

    gc.collect() # garbage collection
    print("\n\u2713 -- Cleared Tensorflow Session.")
    print("\u2713 -- Collected Garbage.")

    try:
        cuda.select_device(0) # release cuda memory back
        cuda.close() # close 
        print("\u2713 -- Released CUDA Memory.")
    except Exception:
        pass
    
    print("\u2713 -- Cleanup Complete.\n\n")


# for per epoch cleanup
class MemoryCleanupCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect() # garbage collect