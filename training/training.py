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
            # Tamanho do conjunto de dados
            DATASET_SIZE = len(labels)
            # Tamanho do conjunto de treinamento e validação
            N_TRAIN = int(0.8 * DATASET_SIZE)
            N_VALIDATION = int(0.2 * DATASET_SIZE)
            BATCH_SIZE = 1
            # Divida o conjunto de dados em treinamento e validação
            validate_ds = dataset.take(N_VALIDATION).cache()
            train_ds = dataset.skip(N_VALIDATION).take(N_TRAIN).cache()
            # Juntar em pacotes de dados e misturar os dados de treinamento
            validate_ds = validate_ds.batch(BATCH_SIZE)
            train_ds = train_ds.shuffle(DATASET_SIZE).batch(BATCH_SIZE)
            print(f"Treinamento: {N_TRAIN}")
            print(f"Validadores: {N_VALIDATION}")
            
            if args.best:
                print('Iniciando procura do melhor modelo!')
                
                tuner = kt.Hyperband(
                    TrainModel,
                    objective='val_accuracy',
                    max_epochs=100,
                    factor=3,
                    max_consecutive_failed_trials=2,
                    directory='models',
                    project_name=f'my-model-{totalModel}'
                )
                tuner.search(
                    train_ds,
                    validation_data=validate_ds,
                    callbacks=[
                        EarlyStopping(monitor='val_accuracy', patience=10, verbose=1),
                        TensorBoard(log_dir='./logs')
                    ],
                    batch_size=1,
                    use_multiprocessing=True
                )
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                best_model = tuner.get_best_models(num_models=1)[0]
                
                print(tuner.get_best_models()[0])
                print(best_model.evaluate(train_ds))
                print("A pesquisa de hiperparâmetros está concluída!")
                
                if tuner.hypermodel is not None:
                    model = tuner.hypermodel.build(best_hps)
                else:
                    print('tuner.hypermodel é None -_-')
                    return
            else:
                model = LoaderModel()

            history = model.fit(
                train_ds,
                validation_data=validate_ds,
                epochs=50,
                batch_size=1,
                callbacks=[
                    ModelCheckpoint(f'models/my-model-{totalModel}/best_model', monitor='val_accuracy', save_best_only=True, mode='auto', verbose=1),
                    EarlyStopping(monitor='val_accuracy', patience=10, verbose=1),
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
            
            print(f"Imagens de treino usadas: {N_TRAIN}")
            print(f"Imagens de Teste usadas: {N_VALIDATION}")

            val_acc_per_epoch = history.history['val_accuracy']
            best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
            print('Best epoch: %d' % (best_epoch))
                
            print(f'Modelo salvo: {datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")}')

asyncio.run(runTraining())
