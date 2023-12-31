import keras
from keras import layers
from keras_tuner import HyperParameters
from keras.optimizers.schedules import InverseTimeDecay

def TrainModel(hp: HyperParameters):
    model = keras.Sequential()
    
    # Camada de Entrada
    model.add(layers.Input(shape=(768, 512, 4)))
    
    loss = hp.Choice('loss', ['BinaryCrossentropy'])
    optimizer = hp.Choice('optimizer', ['Adam'])
    activation = hp.Choice("activation", ["relu"])
    activation_end = hp.Choice("activation_end", ["sigmoid"])
    pooling = hp.Choice('pooling', ['MaxPooling2D', 'AveragePooling2D'])
    learning_rate = hp.Choice('learning_rate', values=[0.001])
    # hp_momentum = hp.Float("momentum", 0.9, 0.99, step=0.03)
    hp_regularizer = hp.Choice("regularizer", ['l1', 'l2'])

    
    hp_filters_1 = hp.Int('filters_1', min_value=32, max_value=256 , step=32)
    hp_filters_2 = hp.Int('filters_2', min_value=16, max_value=128 , step=16)
    hp_filters_3 = hp.Int('filters_3', min_value=8, max_value=64 , step=8)
    hp_kernel_size = hp.Int('kernel_size', min_value=3, max_value=7, step=2)
    # hp_dropout = hp.Float('dropout_rate', 0.2, 0.5, step=0.1)

    # Downsampling
    model.add(layers.Conv2D(filters=hp_filters_1, kernel_size=hp_kernel_size, padding='same', kernel_regularizer=hp_regularizer))
    model.add(layers.Activation(activation=activation))
    model.add(getattr(layers, pooling)(pool_size=(2, 2), strides=(1, 1), padding='same')) # type: ignore   
    model.add(layers.BatchNormalization()) # type: ignore
    
    model.add(layers.Conv2D(filters=hp_filters_2, kernel_size=hp_kernel_size, padding='same', kernel_regularizer=hp_regularizer))
    model.add(layers.Activation(activation=activation))
    model.add(getattr(layers, pooling)(pool_size=(2, 2), strides=(1, 1), padding='same')) # type: ignore
    model.add(layers.BatchNormalization()) # type: ignore
    
    model.add(layers.Conv2D(filters=hp_filters_3, kernel_size=hp_kernel_size, padding='same', kernel_regularizer=hp_regularizer))
    model.add(layers.Activation(activation=activation))
    model.add(getattr(layers, pooling)(pool_size=(2, 2), strides=(1, 1), padding='same')) # type: ignore  
    model.add(layers.BatchNormalization()) # type: ignore
    
    # Camada de saída
    model.add(layers.Conv2D(filters=4, kernel_size=1))
    model.add(layers.Activation(activation=activation_end))
    model.compile(
        loss=getattr(keras.losses, loss)(), # type: ignore
        optimizer = getattr(keras.optimizers, optimizer)(learning_rate), # type: ignore
        metrics=['accuracy']
    )


    return model

def LoaderModel():
    model = keras.Sequential()
    
    # Camada de Entrada
    model.add(layers.Input(shape=(768, 512, 4)))

    # Downsampling
    model.add(layers.Conv2D(filters=96, kernel_size=3, padding='same', kernel_regularizer='l1_l2'))
    model.add(layers.Activation(activation='relu'))
    # model.add(layers.Dropout(rate=hp_dropout))    
    model.add(layers.BatchNormalization(momentum=0.9)) # type: ignore
    
    model.add(layers.Conv2D(filters=32, kernel_size=3, padding='same', kernel_regularizer='l1_l2'))
    model.add(layers.Activation(activation='relu'))
    # model.add(layers.Dropout(rate=hp_dropout))   
    model.add(layers.BatchNormalization(momentum=0.9)) # type: ignore
    
    # Camada de saída
    model.add(layers.Conv2D(filters=4, kernel_size=1))
    model.add(layers.Activation(activation='relu'))
    model.compile(
        loss=keras.losses.BinaryCrossentropy(), # type: ignore
        optimizer = keras.optimizers.Adam(0.001), # type: ignore
        metrics=['accuracy']
    )


    return model