import argparse
import asyncio

import matplotlib.pyplot as plt
from datetime import datetime
import json

import tensorflowjs as tfjs
import tensorflow as tf

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, TerminateOnNaN, ReduceLROnPlateau
from keras import backend as K
import keras_tuner as kt
import keras

from unet.model import FindModel, LoaderModel
from functions.getData import DataLoader
from unet.tensor import TensorLoader


async def runTraining():
    # Starting
    parser = argparse.ArgumentParser(description='Treinamento de modelo')
    parser.add_argument('--best', action='store_true', help='Se o treinamento deve achar o melhor resultado')
    parser.add_argument('--model', type=str, help='Modelo')
    parser.add_argument(
        "--unet",
        action="store_const",
        const=None,
        default=None,
        help="Use U-Net architecture",
    )
    args = parser.parse_args()

    # Aumentar artificalmente a memoria vRam da GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Use apenas a memoria necessaria
            # for gpu in gpus:
            #     tf.config.experimental.set_memory_growth(gpu, True)
            # Limite um certa quantia de memoria
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=7168)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
            
    # Debugging
    # tf.debugging.set_log_device_placement(True)
    
    seed = 319
    tf.random.set_seed(seed)
    
    print("TensorFlow version: ", tf.version)
    device_name = tf.test.gpu_device_name()
    if not device_name:
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))
    
    # Optimization
    policy = keras.mixed_precision.Policy('mixed_float16') # Calculos em float16
    keras.mixed_precision.set_global_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)
    
    tf.config.set_soft_device_placement(True) # Ativar cominicação entre CPU e GPU!

    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tf.profiler.experimental.Profile(logdir=logs)

    # Loader Files
    markDir = 'dados_cache/treino/train'
    loaderFiles = DataLoader()
    imagens, mascaras = await loaderFiles.LoadFiles(markDir, onlyPath=True) or ([], [])
    
    if (imagens is [] or mascaras is []): return print('Nenhum dado carregado!')
    if not isinstance(imagens, list) and not isinstance(imagens[0], str):
        print('Os dados recebidos de imagens e mascaras, são incompativeis!')
        return

    totalModel = loaderFiles.countFolders('models')
    
    image_filenames = tf.constant(imagens)
    masks_filenames = tf.constant(mascaras)

    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))
    
    def process_path(image_path, mask_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=4) # type: ignore
        img = tf.cast(img, tf.float32) / tf.constant(255, dtype=tf.float32)

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=4) # type: ignore
        mask = tf.cast(mask, tf.float32) / tf.constant(255, dtype=tf.float32)
        
        print(tf.math.reduce_min(img), tf.math.reduce_max(img))
        print(tf.math.reduce_min(mask), tf.math.reduce_max(mask))
        # mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
        return img, mask

    def preprocess(image, mask):
        input_image = tf.image.resize(image, (768, 512), method='nearest')
        input_mask = tf.image.resize(mask, (768, 512), method='nearest')
        
        return input_image, input_mask

    image_ds = dataset.map(process_path)
    processed_image_ds = image_ds.map(preprocess)
    def display(display_list):
        plt.figure(figsize=(15, 15))

        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(keras.preprocessing.image.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()
        
    for image, mask in processed_image_ds.take(2):
        sample_image, sample_mask = image, mask
        print(image.shape)
        print(mask.shape)
        display([sample_image, sample_mask])
        
    EPOCHS = 10000
    BUFFER_SIZE = len(imagens)
    BATCH_SIZE = 8
    N_TRAIN = int(0.9 * BUFFER_SIZE)
    N_VALIDATION = int(0.1 * BUFFER_SIZE)
    
    processed_image_ds.shuffle(BUFFER_SIZE)
    validate_ds = processed_image_ds.take(N_VALIDATION).batch(BATCH_SIZE).cache()
    train_ds = processed_image_ds.skip(N_VALIDATION).take(N_TRAIN).batch(BATCH_SIZE).cache()
    print(processed_image_ds.element_spec)
    print(f"Treinamento: {N_TRAIN}")
    print(f"Validadores: {N_VALIDATION}")

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
            max_epochs=EPOCHS,
            factor=5,
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
                ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1), # type: ignore
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
        epochs=EPOCHS,
        batch_size=1,
        callbacks=[
            ModelCheckpoint(f'models/my-model-{totalModel}/best_model', monitor='val_accuracy', save_best_only=True, mode='auto', verbose=1),
            EarlyStopping(monitor='val_accuracy', patience=20, verbose=1),
            TensorBoard(log_dir=logs, histogram_freq=1, embeddings_freq=1),
            TerminateOnNaN(),
            ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1), # type: ignore
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