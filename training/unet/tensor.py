import tensorflow as tf
from PIL import Image
import numpy as np
from memory_profiler import profile
from tqdm import tqdm

class TensorLoader:
    def convert_to_tensor(self, inputs: list[str], labels: list[str]):
        # @profile
        def decode_images(imgPath: str):
                #     Tipo     |      Dimensões     | Uso de memória (MiB)
                # -------------|--------------------|--------
                # decode_img_1 | (387, 512, 256, 4) | 5458.3
                # decode_img_2 | (386, 512, 256, 4) | 5726.7
                # decode_img_3 | (386, 512, 256, 4) | 7007.0
                # decode_img_4 | (387, 512, 256, 4) | 5436.7

                # decode_img_1 = tf.image.decode_image(tf.io.read_file(imgPath), channels=4, dtype=tf.dtypes.float32)
                # decode_img_2 = tf.image.decode_image(contents=imgPath, channels=4, dtype=tf.dtypes.float32)
                # decode_img_3 = keras.utils.load_img(path=imgPath, color_mode='rgba')
                # decode_img_3 = tf.image.convert_image_dtype(decode_img_3, tf.float32)
                decode_img_4 = np.load(imgPath)
                decode_img_4 = decode_img_4['arr_0']

                # resize_img = tf.image.resize(decode_img, [512, 256])
                
                # normalized = tf.cast(decode_img, tf.float32) / 255.0 # type: ignore
                # normalized = tf.image.per_image_standardization(decode_img)
                # print(tf.reduce_min(decode_img_4), tf.reduce_max(decode_img_4))

                # Optional saving for visualization:
                # Best for save image: keras.preprocessing.image.save_img(f"logs/resized-{datetime.datetime.now().timestamp()}-{type}-.png", normalized)
                # keras.preprocessing.image.save_img(f"logs/resized-{datetime.datetime.now().timestamp()}-{type}-.png", decode_img_4)
                # <---- Legacy ---->
                # Teste 1: img_array = tf.cast(tf.clip_by_value(normalized, 0, 1) * 255, tf.uint8).numpy()
                # Teste 2: img_array = (normalized.numpy() * 255).astype(np.uint8)
                # Teste 3: img_array = keras.utils.img_to_array(normalized, dtype='float32')
                # Save: Image.fromarray(img_array, mode="RGBA").save(f"logs/resized-{datetime.datetime.now().timestamp()}-{type}-.png")
                return decode_img_4
        # @profile
        def processImages(imgList):
            #     Tipo     |      Dimensões     | Uso de memória (MiB)
            # -------------|--------------------|--------
            # decode_img_1 | (387, 512, 256, 4) | 7780.8
            # decode_img_2 | (386, 512, 256, 4) | 8043.0
            # decode_img_4 | (387, 512, 256, 4) | 7758.4
            decoded_images = [tf.expand_dims(decode_images(img), axis=0) for img in tqdm(imgList)]
            decoded_images = tf.concat(decoded_images, 0)
            return decoded_images

        with tf.device('/CPU:0'): # type: ignore
            print('Carregando Imagens...')
            input_resized = processImages(inputs)
            print('Carregando Mascaras...')
            label_resized = processImages(labels)

            return input_resized, label_resized
