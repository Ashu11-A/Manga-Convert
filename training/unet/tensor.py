import tensorflow as tf
from PIL import Image
import numpy as np
from memory_profiler import profile
from tqdm import tqdm
import datetime
import keras

class TensorLoader:
    def convert_to_tensor(self, imagesPath: list[str]):
        # @profile
        @tf.function
        def decode_images(imgPath: tuple[str, str]):
                input_tensor = keras.utils.load_img(path=imgPath, color_mode='rgba')
                # output_tensor = keras.utils.load_img(path=imgPath[1], color_mode='rgba')

                # input_bytes = tf.io.read_file(imgPath[0])
                # output_bytes = tf.io.read_file(imgPath[1])
                
                # decode_input_1 = tf.image.decode_image(input_bytes, channels=4, dtype=tf.dtypes.float32) / 255.0 # type: ignore
                # decode_output_1 = tf.image.decode_image(output_bytes, channels=4, dtype=tf.dtypes.float32) / 255.0 # type: ignore
                

                # decode_input_2 = tf.image.convert_image_dtype(input_bytes, tf.float32)
                # decode_output_2 = tf.image.convert_image_dtype(output_bytes, tf.float32)

                # <--- Muito Lent0 --->
                # decode_input_3 = np.load(imgPath[0])
                # decode_input_3 = decode_input_3['arr_0']
                # decode_output_3 = np.load(imgPath[1])
                # decode_output_3 = decode_output_3['arr_0']
                
                decode_input_4 = tf.cast(input_tensor, tf.float32) / 255.0 # type: ignore
                # decode_output_4 = tf.cast(output_tensor, tf.float32) / 255.0 # type: ignore

                # resize_img = tf.image.resize(decode_img_1, [768, 512])
                
                # normalized = tf.cast(decode_img_1, tf.float32) / 255.0 # type: ignore
                # normalized = tf.image.per_image_standardization(decode_img_1)
                # print(tf.reduce_min(decode_img_4), tf.reduce_max(decode_img_4))

                # Optional saving for visualization:
                # keras.preprocessing.image.save_img(f"logs/resized-{datetime.datetime.now().timestamp()}-{type}-.png", normalized)

                # <---- Legacy ---->
                # Teste 1: img_array = tf.cast(tf.clip_by_value(normalized, 0, 1) * 255, tf.uint8).numpy()
                # Teste 2: img_array = (normalized.numpy() * 255).astype(np.uint8)
                # Teste 3: img_array = keras.utils.img_to_array(normalized, dtype='float32')
                # Save: Image.fromarray(img_array, mode="RGBA").save(f"logs/resized-{datetime.datetime.now().timestamp()}-{type}-.png")
                return decode_input_4
        # @profile
        @tf.function
        def processImages(imgList):
            decoded_images = [tf.expand_dims(decode_images(img), axis=0) for img in tqdm(imgList)]
            decoded_images = tf.concat(decoded_images, 0)
            print('Concatenate Terminado...')
            return decoded_images

        with tf.device('/CPU:0'): # type: ignore
            print('Carregando Imagens...')
            porcessedImages = processImages(imagesPath)

            return porcessedImages