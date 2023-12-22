import tensorflow as tf
import keras
from keras import layers
import asyncio
from model.getData import DataLoader
from model.tensor import TensorLoader
import json
import os
from datetime import datetime

async def runTraining():
    markDir: str = 'dados/treino/mark'
    loaderFiles = DataLoader()
    
    folters = loaderFiles.countFolders('models')
    
    print(folters)

    files = await loaderFiles.LoadFiles(markDir)
    if files is not None:
        imagens = files['imagens']
        mascaras = files['mascaras']
        
        if len(imagens) == 0 and len(mascaras) == 0:
            print('Nenhum dado carregado!')
            return
        loaderTensor = TensorLoader()
        tensor = loaderTensor.convertToTensor(inputs=imagens, labels=mascaras)
        inputs = tensor['inputs']
        labels = tensor['labels']
        
        model = keras.Sequential (
            [
                layers.InputLayer(input_shape=[1536, 1024, 4]),
                layers.Dense(units=64, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(units=32, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(units=16, activation='relu'),
                layers.Dense(units=4, activation='sigmoid'),
            ]
        )
        
        model.compile(
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(),
            metrics=keras.metrics.Accuracy()
        )
        
        model.summary()
        
        result = await model.fit(
            x=inputs,
            y=labels,
            batch_size=1,
            epochs=50,
            use_multiprocessing=True
            )
        print('Precisão final: ', result.history)

        totalModal = len([name for name in os.listdir('models') if os.path.isdir(os.path.join('models', name))])
        
        model.save(f'file://models/my-model-{totalModal}')
        
        with open(f'models/my-model-{totalModal}/data.json', 'w') as f:
            json.dump({
                'epochs': result.epoch,
                'history': result.history,
                'data': result.validationData,
                'params': result.params
            }, f)
            
        print(f'Modelo salvo: {datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")}')
            
        prediction = model.predict(tf.random.normal([1, 1536, 1024, 4]))
        print(prediction)
    
asyncio.run(runTraining())