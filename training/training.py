import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
import keras
import asyncio
from model.getData import DataLoader
from model.tensor import TensorLoader
from model.model import LoaderModel
import json
from datetime import datetime
import numpy as np

async def runTraining():
    markDir: str = 'dados/treino/original'
    loaderFiles = DataLoader()
    loaderTensor = TensorLoader()

    files = await loaderFiles.LoadFiles(markDir)
    if files is not None:
        imagens = files['imagens']
        mascaras = files['mascaras']
        
        if len(imagens) == 0 and len(mascaras) == 0:
            print('Nenhum dado carregado!')
            return
        dataset = loaderTensor.convertToTensor(inputs=imagens, labels=mascaras)

        model = LoaderModel()
        
        result = model.fit(
            dataset,
            batch_size=1,
            epochs=50,
            callbacks=keras.callbacks.EarlyStopping(monitor='accuracy', patience=5),
            use_multiprocessing=True
            )
        print('Precisão final: ', result.history)

        totalModel = loaderFiles.countFolders('models')

        keras.saving.save_model(
            model,
            filepath=f'models/my-model-{totalModel}',
            overwrite=True
        )
        
        tfjs.converters.convert_tf_saved_model(
            saved_model_dir=f'models/my-model-{totalModel}',
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