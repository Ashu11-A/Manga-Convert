
import tensorflow as tf

class TensorLoader:
    def convertToTensor(self, inputs: list[bytes], labels: list[bytes]):
         # <-- Faz o redimencionamento da imagens -->
        def resizeImages(img: bytes):
            decodeImg = tf.io.decode_png(img)
            tensorImg = tf.convert_to_tensor(decodeImg)
            return tf.image.resize(tensorImg, [768, 512])

        # Criar Dataset
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        
        # Redimencionar as imagens
        dataset = dataset.map(lambda img, mask: (resizeImages(img), resizeImages(mask)))
        
        # Deixa os dados embaralhados para inpedir viéses
        dataset = dataset.shuffle(buffer_size=100)

        def normalize(inputTensor, labelTensor):
            inputMax = tf.reduce_max(inputTensor)
            inputMin = tf.reduce_min(inputTensor)
            
            labelMax = tf.reduce_max(labelTensor)
            labelMin = tf.reduce_min(labelTensor)
            
            # <-- Normaliza os tensores para o espectro [0, 1] -->
            normalizedInputs = (inputTensor - inputMin) / (inputMax - inputMin)
            normalizedLabels = (labelTensor - labelMin) / (labelMax - labelMin)
            return normalizedInputs, normalizedLabels
        
        dataset = dataset.map(normalize)
        dataset = dataset.batch(batch_size=1)
        
        return dataset