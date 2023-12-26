import asyncio
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
import keras_tuner as kt

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
        
        if result is not None:
            inputs = result['inputs']
            labels = result['labels']
            
            K.clear_session()
            
            tuner = kt.Hyperband(
                LoaderModel,
                objective='val_accuracy',
                max_epochs=50,
                factor=3,
                project_name=f'models/my-model-{totalModel}'
            )
            stop_early = keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)
            
            tuner.search(inputs, labels, epochs=20, validation_split=0.2, callbacks=[stop_early], batch_size=1, use_multiprocessing=True)
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            
            model = tuner.hypermodel.build(best_hps)
            history = model.fit(
                inputs,
                labels,
                epochs=50,
                validation_split=0.2,
                batch_size=1,
                callbacks=[
                    ModelCheckpoint(f'models/my-model-{totalModel}/best_model', verbose=1, monitor='accuracy', save_best_only=True, mode='auto'),
                    EarlyStopping(monitor='accuracy', patience=5),
                    TensorBoard(log_dir='./logs')
                ],
                use_multiprocessing=True
            )
            
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
                    'epochs': history.epoch,
                    'history': history.history,
                    'data': history.validation_data,
                    'params': history.params
                }, dataFile)
                
            # print(tuner.get_best_models()[0])
            
            print(f"""
                  A pesquisa de hiperparâmetros está concluída!
                  O número ótimo de unidades na primeira densely-connected é {best_hps.get('units1')}.
                  A segunda camada do densely-connected foi de {best_hps.get('units2')}.
                  E por ultimo, a taxa de optimal learning rate para o optimizador é {best_hps.get('learning_rate')}.
            """)

            val_acc_per_epoch = history.history['val_accuracy']
            best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
            print('Best epoch: %d' % (best_epoch))
                
            print(f'Modelo salvo: {datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")}')

asyncio.run(runTraining())
