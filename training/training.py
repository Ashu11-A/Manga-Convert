import asyncio
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
import keras
from model.getData import DataLoader
from model.tensor import TensorLoader
from model.model import LoaderModel
import json
from datetime import datetime
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import io

async def runTraining():
    markDir = 'dados/treino/train'
    loaderFiles = DataLoader()
    loaderTensor = TensorLoader()
    print("TensorFlow version: ", tf.version)
    device_name = tf.test.gpu_device_name()
    if not device_name:
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    files = await loaderFiles.LoadFiles(markDir)
    if files is not None:
        imagens = files['images']
        mascaras = files['masks']

        if len(imagens) == 0 and len(mascaras) == 0:
            print('Nenhum dado carregado!')
            return

        totalModel = loaderFiles.countFolders('models')
        result = loaderTensor.convert_to_tensor(inputs=imagens, labels=mascaras)
        model = LoaderModel()
        
        if result is not None:
            inputs = result['inputs']
            labels = result['labels']
            
            K.clear_session()

            result = model.fit(
                inputs,
                labels,
                epochs=50,
                shuffle=True,
                callbacks=[
                    ModelCheckpoint(f'models/my-model-{totalModel}/best_model', verbose=1, monitor='accuracy', save_best_only=True, mode='auto'),
                    EarlyStopping(monitor='accuracy', patience=5),
                    TensorBoard(log_dir='./logs')
                ],
                use_multiprocessing=True
            )

            print('Precisão final: ', result.history)

            keras.models.save_model(
                model,
                filepath=f'models/my-model-{totalModel}',
                overwrite=True
            )

            tfjs.converters.convert_tf_saved_model(
                saved_model_dir=f'models/my-model-{totalModel}/best_model',
                output_dir=f'models/my-model-{totalModel}'
            )

            with open(f'models/my-model-{totalModel}/data.json', 'w') as dataFile:
                json.dump({
                    'epochs': result.epoch,
                    'history': result.history,
                    'data': result.validation_data,
                    'params': result.params
                }, dataFile)

            print(f'Modelo salvo: {datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")}')
    
asyncio.run(runTraining())
