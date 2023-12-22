
import tensorflow as tf

class TensorLoader:
    def convertToTensor(self, inputs: list[bytes], labels: list[bytes]):
         # <-- Faz o redimencionamento da imagens -->
        def resizeImages(img: bytes):   
            decodeImg = tf.io.decode_image(img)
            tensorImg = tf.convert_to_tensor(decodeImg)
            return tf.image.resize(tensorImg, [1536, 1024])

        inputResized = list(map(resizeImages, inputs))
        labelResized = list(map(resizeImages, labels))

        inputTensor = tf.stack(inputResized)
        labelsTensor = tf.stack(labelResized)
        
        # <-- Embaralhamento -->
        indices = tf.range(start=0, limit=inputTensor.shape[0], dtype=tf.int32)
        shuffledIndices = tf.random.shuffle(indices)
        
        inputTensor = tf.gather(inputTensor, shuffledIndices)
        labelsTensor = tf.gather(labelsTensor, shuffledIndices)

        inputMax = tf.reduce_max(inputTensor)
        labelMax = tf.reduce_max(inputTensor)
        inputMin = tf.reduce_min(inputTensor)
        labelMin = tf.reduce_min(labelsTensor)
        
         # <-- Normaliza os tensores para o espectro [0, 1] -->
        normalized_inputs = (inputTensor - inputMin) / (inputMax - inputMin)
        normalized_labels = (labelsTensor - labelMin) / (labelMax - labelMin)
        
        return {
            'inputs': normalized_inputs,
            'labels': normalized_labels,
            'inputMax': inputMax,
            'inputMin': inputMin,
            'labelMax': labelMax,
            'labelMin': labelMin,
        }