from typing import List
import tensorflow as tf
import numpy as np
import keras

class TensorLoader:
    def convert_to_tensor(self, inputs: List[bytes], labels: List[bytes]):

        def resize_images(img: bytes):
            decode_img = tf.image.decode_image(img, channels = 4)
            resize_img = tf.image.resize(decode_img, [768, 512])
            return resize_img

        input_resized = tf.stack([resize_images(img) for img in inputs])
        label_resized = tf.stack([resize_images(img) for img in labels])

        # Converting to [0, 1]
        input_max = tf.reduce_max(input_resized)
        input_min = tf.reduce_min(input_resized)
        label_max = tf.reduce_max(label_resized)
        label_min = tf.reduce_min(label_resized)

        normalized_inputs = (input_resized - input_min) / (input_max - input_min)
        normalized_labels = (label_resized - label_min) / (label_max - label_min)

        return {
            'inputs': normalized_inputs,
            'labels': normalized_labels,
            'input_max': input_max,
            'input_min': input_min,
            'label_max': label_max,
            'label_min': label_min
        }
