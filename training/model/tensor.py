from typing import List
import tensorflow as tf
from tensorflow_addons.image import transform as image_transform

class TensorLoader:
    def convert_to_tensor(self, inputs: List[bytes], labels: List[bytes]):
        tf.config.set_soft_device_placement(True)
        with tf.device('/gpu:0'): # type: ignore

            def resize_images(img: bytes):
                    decode_img = tf.image.decode_image(contents=img, channels=4, dtype=tf.dtypes.float16) # GPU deve usar float32, CPU uint8
                    resize_img = tf.image.resize(decode_img, [768, 512])
                    # normalized = tf.image.per_image_standardization(resize_img)
                    # print(tf.reduce_min(resize_img), tf.reduce_max(resize_img))

                    # Optional saving for visualization:
                    # img_array = tf.cast(tf.clip_by_value(resize_img, 0, 1) * 255, tf.uint8).numpy()
                    # Image.fromarray(img_array).save(f"logs/resized-{number}-{type}-.png")
                    return resize_img
            input_resized = tf.stack([resize_images(img) for img in inputs])
            label_resized = tf.stack([resize_images(img) for img in labels])
            
            print(input_resized.shape)
            print(label_resized.shape)
            print(input_resized.device)
            print(input_resized.device)

            return {
                'inputs': input_resized,
                'labels': label_resized
            }
