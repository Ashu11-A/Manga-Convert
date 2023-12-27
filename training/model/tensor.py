from typing import List
import tensorflow as tf
import numpy as np

class TensorLoader:
    def convert_to_tensor(self, inputs: List[bytes], labels: List[bytes]):

        def resize_images(img: bytes):
            decode_img = tf.image.decode_image(img, channels=4, dtype=tf.dtypes.float32) # GPU deve usar float32, CPU uint8
            resize_img = tf.image.resize(decode_img, [768, 512])
            # normalized = tf.image.per_image_standardization(resize_img)
            # print(tf.reduce_min(resize_img), tf.reduce_max(resize_img))
            return resize_img

        input_resized = tf.stack([resize_images(img) for img in inputs])
        label_resized = tf.stack([resize_images(img) for img in labels])

        return {
            'inputs': input_resized,
            'labels': label_resized
        }
