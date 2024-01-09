import asyncio
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
import keras
import json
from datetime import datetime
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, TerminateOnNaN, LambdaCallback, ReduceLROnPlateau, ProgbarLogger, BackupAndRestore
import keras_tuner as kt
import argparse
import matplotlib.pyplot as plt

from functions.getData import DataLoader
from unet.tensor import TensorLoader
from unet.model import FindModel, LoaderModel

async def runTraining():
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
            
    # Debugging
    tf.debugging.set_log_device_placement(True)
    
    seed = 319
    tf.random.set_seed(seed)
    
    print("TensorFlow version: ", tf.version)
    device_name = tf.test.gpu_device_name()
    if not device_name:
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))
    
    # Optimization
    # policy = keras.mixed_precision.Policy('mixed_float16') # Calculos em float16
    # keras.mixed_precision.set_global_policy(policy)
    # print('Compute dtype: %s' % policy.compute_dtype)
    # print('Variable dtype: %s' % policy.variable_dtype)
    
    tf.config.set_soft_device_placement(True) # Ativar cominicação entre CPU e GPU!

    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tf.profiler.experimental.Profile(logdir=logs)


    markDir = 'dados_cache/treino/train'
    loaderFiles = DataLoader()
    loaderTensor = TensorLoader()
    parser = argparse.ArgumentParser(description='Treinamento de modelo')
    parser.add_argument('--best', action='store_true', help='Se o treinamento deve achar o melhor resultado')
    parser.add_argument('--model', type=str, help='Modelo')
    args = parser.parse_args()

    imagens, mascaras = await loaderFiles.LoadFiles(markDir, onlyPath=True) or ([], [])
    
    if (imagens is [] or mascaras is []): return print('Nenhum dado carregado!')

    if not isinstance(imagens, list) and not isinstance(imagens[0], str):
        print('Os dados recebidos de imagens e mascaras, são incompativeis!')
        return

    totalModel = loaderFiles.countFolders('models')
    inputs, labels = loaderTensor.convert_to_tensor(inputs=imagens, labels=mascaras) # type: ignore
    
    
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    # Tamanho do conjunto de dados
    DATASET_SIZE = len(imagens)
    # Embaralha as coisas
    dataset = dataset.shuffle(DATASET_SIZE)
    # Tamanho do conjunto de treinamento e validação
    N_TRAIN = int(0.8 * DATASET_SIZE)
    N_VALIDATION = int(0.2 * DATASET_SIZE)
    BATCH_SIZE = 4

    # Divida o conjunto de dados em treinamento e validação
    validate_ds = dataset.take(N_VALIDATION).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    train_ds = dataset.skip(N_VALIDATION).take(N_TRAIN).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    print(f"Treinamento: {N_TRAIN}")
    print(f"Validadores: {N_VALIDATION}")
    
    def display(display_list):
        plt.figure(figsize=(15, 15))
        title = ["Input Image", "True Mask"]
        
        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.figure(facecolor='black')
            plt.title(title[i])
            plt.imshow(keras.utils.array_to_img(display_list[i]))
            plt.axis('off')
        plt.savefig('./teste.png')
    sample_batch = next(iter(train_ds))
    random_index = np.random.choice(sample_batch[0].shape[0])
    sample_image, sample_mask = sample_batch[0][random_index], sample_batch[1][random_index]
    display([sample_image, sample_mask])

    if args.model:
        print(F'Retreinando o Modelo: {args.model}')
        model = keras.models.load_model(F"models/my-model-{args.model}")
        if model is None:
            print('Não existe um checkpoint!')
            return
    elif args.best:
        print('Iniciando procura do melhor modelo!')
        
        tuner = kt.Hyperband(
            FindModel,
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
                EarlyStopping(monitor='val_accuracy', patience=25, verbose=1),
                TensorBoard(log_dir=logs, histogram_freq=1, profile_batch=2),
                TerminateOnNaN(),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, cooldown=5, min_lr=0.000000001, verbose=1), # type: ignore
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
            EarlyStopping(monitor='val_accuracy', patience=25, verbose=1),
            TensorBoard(log_dir=logs, histogram_freq=1, profile_batch=2),
            TerminateOnNaN(),
            ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, cooldown=5, min_lr=0.000000001), # type: ignore
            BackupAndRestore(f'models/my-model-{totalModel}/checkpoints')
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
