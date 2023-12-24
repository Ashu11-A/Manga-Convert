
import tensorflow as tf

class TensorLoader:
    def convertToTensor(self, inputs: list[bytes], labels: list[bytes]):
         # <-- Faz o redimencionamento da imagens -->
        def resizeImages(img: bytes):
            decodeImg = tf.io.decode_png(img, channels = 4)
            convertImg = tf.image.resize(decodeImg, [768, 512])
            
            minVal = tf.reduce_min(convertImg)
            maxVal = tf.reduce_max(convertImg)
            
            if minVal == maxVal:
                return convertImg
            else:
                return (convertImg - minVal) / (maxVal - minVal)

        # Criar Dataset
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        
        # Redimencionar as imagens
        dataset = dataset.map(lambda img, mask: (resizeImages(img), resizeImages(mask)))
        
        # Deixa os dados embaralhados para inpedir viéses
        dataset = dataset.shuffle(buffer_size=100).batch(batch_size=1)
        
        return dataset