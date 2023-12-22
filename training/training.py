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
        dataset = loaderTensor.convertToTensor(inputs=imagens, labels=mascaras)
        
        model = keras.Sequential (
            [
                layers.InputLayer(input_shape=[768, 512, 4]),
                layers.Dense(units=128, activation='relu'),
                layers.Dropout(0.1),
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
            metrics=keras.metrics.Accuracy(),
            run_eagerly=True
        )
        
        model.summary()
        
        result = await model.fit(
            dataset,
            batch_size=1,
            epochs=50,
            use_multiprocessing=True
            )
        print('Precisão final: ', result.history)

        totalModal = len([name for name in os.listdir('models') if os.path.isdir(os.path.join('models', name))])

        keras.saving.save_model(
            model,
            filepath=f'models/my-model-{totalModal}',
            overwrite=True
        )
        
        with open(f'models/my-model-{totalModal}/model.json') as jsonFile:
            jsonFile.write(model.to_json())
        with open(f'models/my-model-{totalModal}/data.json', 'w') as dataFile:
            json.dump({
                'epochs': result.epoch,
                'history': result.history,
                'data': result.validationData,
                'params': result.params
            }, dataFile)
            
        print(f'Modelo salvo: {datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")}')
            
        prediction = model.predict(tf.random.normal([1, 768, 512, 4]))
        print(prediction)
    
asyncio.run(runTraining())