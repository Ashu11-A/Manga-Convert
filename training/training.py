import asyncio
import tensorflow as tf
import tensorflowjs as tfjs
import keras
from model.getData import DataLoader
from model.tensor import TensorLoader
from model.model import TrainModel, LoaderModel
import json
from datetime import datetime
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import keras_tuner as kt
import argparse


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
            
            parser = argparse.ArgumentParser(description='Treinamento de modelo')
            parser.add_argument('--best', action='store_true', help='Se o treinamento deve achar o melhor resultado')
            args = parser.parse_args()
            
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
            # Suponha que DATASET_SIZE é o tamanho do seu conjunto de dados
            DATASET_SIZE = len(inputs)
            # Defina o tamanho do conjunto de treinamento e validação
            train_size = int(0.8 * DATASET_SIZE)
            val_size = int(0.2 * DATASET_SIZE)
            # Embaralhe o conjunto de dados
            dataset = dataset.shuffle(DATASET_SIZE)
            # Divida o conjunto de dados em treinamento e validação
            train_dataset = dataset.take(train_size).batch(1)
            val_dataset = dataset.skip(train_size).take(val_size).batch(1)
            
            if args.best:
                print('Iniciando procura do melhor modelo!')
                tuner = kt.Hyperband(
                    TrainModel,
                    objective='val_accuracy',
                    max_epochs=20,
                    factor=3,
                    directory='models',
                    project_name=f'my-model-{totalModel}'
                )
                stop_early = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
                tuner.search(train_dataset, validation_data=val_dataset, epochs=20, callbacks=[stop_early], batch_size=1, use_multiprocessing=True)
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                
                print(tuner.get_best_models()[0])
                print(f"""
                      A pesquisa de hiperparâmetros está concluída!
                      O número ótimo de unidades: 1º {best_hps.get('units1')}, 2º {best_hps.get('units2')}, 3º {best_hps.get('units3')}.
                      E por ultimo, a taxa de optimal learning rate para o optimizador é {best_hps.get('learning_rate')}.
                """)
                
                if tuner.hypermodel is not None:
                    model = tuner.hypermodel.build(best_hps)
                else:
                    print('tuner.hypermodel é None -_-')
                    return
            else:
                model = LoaderModel()

            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=50,
                batch_size=1,
                callbacks=[
                    ModelCheckpoint(f'models/my-model-{totalModel}/best_model', monitor='val_accuracy', save_best_only=True, mode='auto'),
                    EarlyStopping(monitor='val_accuracy', patience=10),
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
            
            print(f"Imagens de treino usadas: {train_size}")
            print(f"Imagens de Teste usadas: {val_size}")

            val_acc_per_epoch = history.history['val_accuracy']
            best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
            print('Best epoch: %d' % (best_epoch))
                
            print(f'Modelo salvo: {datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")}')

asyncio.run(runTraining())
