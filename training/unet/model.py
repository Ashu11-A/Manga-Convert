import numbers
from typing import List, Union
import keras as keras
from keras import layers
from keras.regularizers import L1
from keras.layers import Conv2D, Input, ReLU, BatchNormalization, concatenate, Dropout, Conv2DTranspose, Conv2DTranspose
from keras_tuner import HyperParameters

def FindModel(hp: HyperParameters): 
    # Camada de Entrada
    loss = hp.Choice('loss', ['BinaryCrossentropy'])
    optimizer = hp.Choice('optimizer', ['Adam'])
    # activation = hp.Choice("activation", ["relu"])
    activation_end = hp.Choice("activation_end", ["sigmoid"])
    pooling = hp.Choice('pooling', ['MaxPooling2D', 'AveragePooling2D'])
    # upscale = hp.Choice('upscale', ['Conv2DTranspose'])
    learning_rate = hp.Choice('learning_rate', values=[0.001])
    kernel_initializer = hp.Choice('kernel_initializer', ['he_normal'])
    kernel_size = hp.Choice('kernel_size', values=[3])
    dropout = hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
    filter = hp.Choice('filter', values=[4, 8, 16])
    
    
    def down_block(x, filters: int, factor: int, use_maxpool=True):
        x = Conv2D(filters * factor, kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = Dropout(dropout)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters * factor, kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = Dropout(dropout)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        if use_maxpool:
            return getattr(layers, str(pooling))((2, 2))(x), x
        return x
    
    def up_block(x, y, filters, factor: int):
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = concatenate([x, y])
        x = Conv2D(filters * factor, kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = Dropout(dropout)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters * factor, kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = Dropout(dropout)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    
    # encode
    input = Input(shape=(768, 512, 4))
    x, temp1 = down_block(input, filter, factor=2) # type: ignore
    x, temp2 = down_block(x, filter, factor=4) # type: ignore
    x, temp3 = down_block(x, filter, factor=8) # type: ignore
    x, temp4 = down_block(x, filter, factor=16) # type: ignore
    
    x = down_block(x, filter, use_maxpool = False, factor=32) # type: ignore
    
    # decode 
    x = up_block(x, temp4, filter, factor=16) # type: ignore
    x = up_block(x, temp3, filter, factor=8) # type: ignore
    x = up_block(x, temp2, filter, factor=4) # type: ignore
    x = up_block(x, temp1, filter, factor=2) # type: ignore
    
    output = Conv2D(4, (1, 1), activation=activation_end, dtype='float32')(x)
    
    model = keras.Model(input, output, name='u-net')
    
    # Camada de saída
    model.compile(
        loss = getattr(keras.losses, str(loss))(),
        optimizer = getattr(keras.optimizers, str(optimizer))(learning_rate),
        metrics=['accuracy']
    )
    model.summary()

    return model

def LoaderModel():
    # ID: 347 - val_accuracy: [0.87415] | filter: [32, 64, 128, 256, 512] - kernel_size: 3 / 0.4
    # ID: 382 - val_accuracy: [0.90806] | filter: [32, 64, 128, 256, 512] - kernel_size: 7 / 0.2
    # ID: 383 - val_accuracy: [0.89926] | filter: [32, 64, 128, 256, 512] - kernel_size: 3 / 0.2
    # ID: 385 - val_accuracy: [0.89182] | filter: [32, 64, 128, 256, 512] - kernel_size: 7 / 0.4
    # ID: 399 - val_accuracy: [0.90410] | filter: [32, 64, 128, 256, 512] - kernel_size: 3 / 0.2
    loss = 'BinaryCrossentropy'
    optimizer = 'Adam'
    learning_rate = 0.001
    kernel_size = 3
    dropout = 0.2
    filter = 16
    
    kernel_initializer = 'he_normal'
    activation_end = 'sigmoid'
    pooling = 'MaxPooling2D'

    def down_block(x, filters: int, factor: int, use_maxpool=True):
        x = Conv2D(filters * factor, kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters * factor, kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        if use_maxpool:
            return getattr(layers, str(pooling))((2, 2))(x), x
        else:
            return x
    
    def up_block(x, y, filters, factor: int):
        x = Conv2DTranspose(filters, kernel_size, strides=(2, 2), padding='same')(x)
        x = concatenate([x, y])
        x = Conv2D(filters * factor, kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters * factor, kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    
    # encode
    input = Input(shape=(768, 512, 4))
    x, temp1 = down_block(input, filter, factor=2) # type: ignore
    x, temp2 = down_block(x, filter, factor=4) # type: ignore
    x, temp3 = down_block(x, filter, factor=8) # type: ignore
    x, temp4 = down_block(x, filter, factor=16) # type: ignore
    
    x = down_block(x, filter, use_maxpool = False, factor=32)
    
    # decode 
    x = up_block(x, temp4, filter, factor=16)
    x = up_block(x, temp3, filter, factor=8)
    x = up_block(x, temp2, filter, factor=4)
    x = up_block(x, temp1, filter, factor=2)
    
    output = Conv2D(4, (1, 1), activation=activation_end, dtype='float32')(x)
    
    model = keras.Model(input, output, name='u-net')
    
    # Camada de saída
    model.compile(
        loss = getattr(keras.losses, str(loss))(),
        optimizer = getattr(keras.optimizers, str(optimizer))(learning_rate),
        metrics=['accuracy']
    )
    model.summary()

    return model