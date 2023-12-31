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
    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Uma tal de "Otimização"
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)
    
    tf.config.set_soft_device_placement(True)
    tf.profiler.experimental.Profile(logdir=logs)
    # Aumentar artificalmente a memoria vRam da GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=7168)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    markDir = 'dados_cache/treino/train'
    loaderFiles = DataLoader()
    loaderTensor = TensorLoader()
    print("TensorFlow version: ", tf.version)
    device_name = tf.test.gpu_device_name()
    if not device_name:
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    imagens, mascaras = await loaderFiles.LoadFiles(markDir, onlyPath=True) or ([], [])
    
    if (imagens is [] or mascaras is []): return print('Nenhum dado carregado!')

    if not isinstance(imagens, list) and not isinstance(imagens[0], str):
        print('Os dados recebidos de imagens e mascaras, são incompativeis!')
        return

    totalModel = loaderFiles.countFolders('models')
    inputs, labels = loaderTensor.convert_to_tensor(inputs=imagens, labels=mascaras) # type: ignore
    
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)
    
    parser = argparse.ArgumentParser(description='Treinamento de modelo')
    parser.add_argument('--best', action='store_true', help='Se o treinamento deve achar o melhor resultado')
    args = parser.parse_args()
    
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    # Tamanho do conjunto de dados
    DATASET_SIZE = len(labels)
    # Embaralha as coisas
    dataset = dataset.shuffle(DATASET_SIZE)
    # Tamanho do conjunto de treinamento e validação
    N_TRAIN = int(0.8 * DATASET_SIZE)
    N_VALIDATION = int(0.2 * DATASET_SIZE)
    BATCH_SIZE = 1
    # Divida o conjunto de dados em treinamento e validação
    validate_ds = dataset.take(N_VALIDATION).cache()
    train_ds = dataset.skip(N_VALIDATION).take(N_TRAIN).cache()
    # Juntar em pacotes de dados e misturar os dados de treinamento
    validate_ds = validate_ds.batch(BATCH_SIZE)
    train_ds = train_ds.batch(BATCH_SIZE)
    print(f"Treinamento: {len(list(train_ds.as_numpy_iterator()))}")
    print(f"Validadores: {len(list(validate_ds.as_numpy_iterator()))}")

    if args.best:
        print('Iniciando procura do melhor modelo!')
        
        tuner = kt.Hyperband(
            TrainModel,
            objective='val_accuracy',
            max_epochs=100,
            factor=3,
            max_consecutive_failed_trials=3,
            directory='models',
            project_name=f'my-model-{totalModel}'
        )
        tuner.search(
            train_ds,
            validation_data=validate_ds,
            callbacks=[
                EarlyStopping(monitor='val_accuracy', patience=20, verbose=1),
                EarlyStopping(monitor='val_loss', patience=20, verbose=1),
                EarlyStopping(monitor='accuracy', patience=20, verbose=1),
                EarlyStopping(monitor='loss', patience=20, verbose=1),
                TensorBoard(log_dir=logs, histogram_freq=1, profile_batch=3)
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
        
    K.clear_session()
    
    history = model.fit(
        train_ds,
        validation_data=validate_ds,
        epochs=100,
        batch_size=1,
        callbacks=[
            ModelCheckpoint(f'models/my-model-{totalModel}/best_model', monitor='val_accuracy', save_best_only=True, mode='auto', verbose=1),
            EarlyStopping(monitor='val_accuracy', patience=20, verbose=1),
            TensorBoard(log_dir=logs, histogram_freq=1, profile_batch=2)
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
