import numpy as np
import tensorflow as tf
import os

class PointHistoryClassifier(object):
    def __init__(self, model_path=None, num_threads=1):
        if model_path is None:
            # Get the directory where this file is located
            model_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(model_dir, 'point_history_classifier.tflite')
            
        print(f"Loading PointHistoryClassifier model from: {model_path}")
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                             num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, point_history):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([point_history], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))

        return result_index 