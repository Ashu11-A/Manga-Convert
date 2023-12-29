import keras
from keras import layers
from keras_tuner import HyperParameters
from keras.optimizers.schedules import InverseTimeDecay

def TrainModel(hp: HyperParameters):
    model = keras.Sequential()
    
    # Camada de Entrada
    model.add(layers.Input(shape=(768, 512, 4)))
    
    loss = hp.Choice('loss', ['BinaryCrossentropy', 'BinaryFocalCrossentropy'])
    optimizer = hp.Choice('optimizer', ['Adam', 'RMSprop'])
    activation = hp.Choice("activation", ["relu", "tanh"])
    activation_end = hp.Choice("activation_end", ["sigmoid"])
    learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001])
    lr_schedule = InverseTimeDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        decay_rate=0.1
    )

    
    hp_filters_1 = hp.Int('filters_1', min_value=32, max_value=256 , step=32)
    hp_filters_2 = hp.Int('filters_2', min_value=32, max_value=256 , step=32)
    hp_filters_3 = hp.Int('filters_3', min_value=32, max_value=256 , step=32)
    hp_kernel_size = hp.Int('kernel_size', min_value=3, max_value=7, step=2)
    # hp_upscale_1 = hp.Int('upscale_1', min_value=4, max_value=128, step=2)
    # hp_upscale_2 = hp.Int('upscale2', min_value=4, max_value=128, step=2)
    hp_momentum = hp.Float("momentum", 0.6, 0.9, step=0.2)
    hp_dropout = hp.Float('dropout_rate', 0.2, 0.5, step=0.1)

    # Downsampling
    model.add(layers.Conv2D(filters=hp_filters_1, kernel_size=hp_kernel_size, activation=activation, padding='same'))
    # model.add(layers.MaxPooling2D((2, 2), strides=2))
    model.add(layers.BatchNormalization(momentum=hp_momentum)) # type: ignore
    model.add(layers.Dropout(rate=hp_dropout))    
    
    model.add(layers.Conv2D(filters=hp_filters_2, kernel_size=hp_kernel_size, activation=activation, padding='same'))
    # model.add(layers.MaxPooling2D((2, 2), strides=2))
    model.add(layers.BatchNormalization(momentum=hp_momentum)) # type: ignore
    model.add(layers.Dropout(rate=hp_dropout))   
    
    model.add(layers.Conv2D(filters=hp_filters_3, kernel_size=hp_kernel_size, activation=activation, padding='same'))
    model.add(layers.BatchNormalization(momentum=hp_momentum)) # type: ignore
    model.add(layers.Dropout(rate=hp_dropout))
    # model.add(layers.MaxPooling2D((2, 2)))
    
    # Upsampling
    # model.add(layers.Conv2DTranspose(filters=hp_upscale_1, kernel_size=(1, 1), activation=activation))
    # model.add(layers.UpSampling2D((2, 2)))
    # model.add(layers.BatchNormalization(momentum=hp_momentum)) # type: ignore

    # model.add(layers.Conv2DTranspose(filters=hp_upscale_2, kernel_size=(1, 1), activation=activation))
    # model.add(layers.UpSampling2D((4, 4)))
    
    # Camada de saída
    model.add(layers.Conv2D(filters=4, kernel_size=1, activation=activation_end))
    model.compile(
        loss=getattr(keras.losses, loss)(), # type: ignore
        optimizer = getattr(keras.optimizers, optimizer)(learning_rate=lr_schedule), # type: ignore
        metrics=['accuracy']
    )


    return model

def LoaderModel():
    model = keras.Sequential()
    
    # Camada de Entrada
    model.add(layers.Input(shape=(768, 512, 4)))

    # Downsampling
    model.add(layers.Conv2D(filters=160, kernel_size=5, activation='relu', padding='same'))
    model.add(layers.BatchNormalization(momentum=0.6)) # type: ignore
    model.add(layers.Dropout(rate=0.3))    
    
    model.add(layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding='same'))
    model.add(layers.BatchNormalization(momentum=0.6)) # type: ignore
    model.add(layers.Dropout(rate=0.3))   
    
    model.add(layers.Conv2D(filters=128, kernel_size=5, activation='relu', padding='same'))
    model.add(layers.BatchNormalization(momentum=0.6)) # type: ignore
    model.add(layers.Dropout(rate=0.3))
    
    # Camada de saída
    model.add(layers.Conv2D(filters=4, kernel_size=1, activation='sigmoid'))
    model.compile(
        loss=keras.losses.BinaryFocalCrossentropy(), # type: ignore
        optimizer = keras.optimizers.Adam(learning_rate=0.01), # type: ignore
        metrics=['accuracy'],
    )
    
    return model