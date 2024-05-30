import keras as keras
from keras import layers
from keras.layers import Conv2D, Input, ReLU, BatchNormalization, concatenate, Dropout, Conv2DTranspose, Conv2DTranspose
from keras_tuner import HyperParameters

def FindModel(hp: HyperParameters): 
    # Camada de Entrada
    loss = hp.Choice('loss', ['BinaryCrossentropy'])
    optimizer = hp.Choice('optimizer', ['Adam'])
    activation = hp.Choice("activation", ["relu"])
    activation_end = hp.Choice("activation_end", ["sigmoid"])
    pooling = hp.Choice('pooling', ['MaxPooling2D'])
    # upscale = hp.Choice('upscale', ['Conv2DTranspose'])
    learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])
    kernel_initializer: str = hp.Choice('kernel_initializer', ['he_normal']) # type: ignore
    kernel_size = hp.Choice('kernel_size', values=[3])
    dropout: int = hp.Float('dropout_rate', 0.1, 0.5, step=0.1) # type: ignore
    filter: int = hp.Choice('filter', values=[4, 8, 16]) # type: ignore
    input = Input(shape=(512 ,320, 3))
    
    def down_block(x, filters: int, dropout_prob: float = 0, use_maxpool=True):
        x = Conv2D(filters, kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        if dropout_prob > 0:
            x = Dropout(dropout_prob)(x)
        if use_maxpool:
            return getattr(layers, str(pooling))((2, 2))(x), x
        return x, x
    
    def up_block(x, y, filters):
        x = Conv2DTranspose(filters, kernel_size, strides=(2, 2), padding='same')(x)
        x = concatenate([x, y], axis=3)
        x = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        return x
    
    # encode
    cblock1 = down_block(input, filter)
    cblock2 = down_block(cblock1[0], filter * 2)
    cblock3 = down_block(cblock2[0], filter * 4)
    cblock4 = down_block(cblock3[0], filter* 8, dropout_prob=dropout)
    
    cblock5 = down_block(cblock4[0], filter * 16, use_maxpool = False, dropout_prob=dropout)
    
    # decode 
    ublock6 = up_block(cblock5[0], cblock4[1], filter * 8)
    ublock7 = up_block(ublock6, cblock3[1], filter * 4)
    ublock8 = up_block(ublock7, cblock2[1], filter * 2)
    ublock9 = up_block(ublock8, cblock1[1], filter)
    
    conv9 = Conv2D(filter,
                3,
                activation='relu',
                padding='same',
                kernel_initializer=f"{kernel_initializer}")(ublock9)
    
    conv10 = Conv2D(4, (1, 1), activation=activation_end, dtype='float32')(conv9)
    
    model = keras.Model(input, conv10, name='u-net')
    
    # Camada de saída
    model.compile(
        loss = getattr(keras.losses, str(loss))(),
        optimizer = getattr(keras.optimizers, str(optimizer))(learning_rate),
        metrics=['accuracy']
    )
    model.summary()

    return model

def LoaderModel():
    input = Input(shape=(512, 320, 3))
    loss = 'BinaryCrossentropy'
    optimizer = 'Adam'
    learning_rate = 0.001
    kernel_size = 3
    dropout = 0.2
    filter = 16
    activation = 'relu'
    
    kernel_initializer = 'he_normal'
    activation_end = 'sigmoid'
    pooling = 'MaxPooling2D'

    def down_block(x, filters: int, dropout_prob: float = 0, use_maxpool=True):
        x = Conv2D(filters, kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        if dropout_prob > 0:
            x = Dropout(dropout_prob)(x)
        if use_maxpool:
            return getattr(layers, str(pooling))((2, 2))(x), x
        return x, x
    
    def up_block(x, y, filters):
        x = Conv2DTranspose(filters, kernel_size, strides=(2, 2), padding='same')(x)
        x = concatenate([x, y], axis=3)
        x = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        return x
    
    # encode
    cblock1 = down_block(input, filter)
    cblock2 = down_block(cblock1[0], filter * 2)
    cblock3 = down_block(cblock2[0], filter * 4)
    cblock4 = down_block(cblock3[0], filter* 8, dropout_prob=dropout)
    
    cblock5 = down_block(cblock4[0], filter * 16, use_maxpool = False, dropout_prob=dropout)
    
    # decode 
    ublock6 = up_block(cblock5[0], cblock4[1], filter * 8)
    ublock7 = up_block(ublock6, cblock3[1], filter * 4)
    ublock8 = up_block(ublock7, cblock2[1], filter * 2)
    ublock9 = up_block(ublock8, cblock1[1], filter)
    
    conv9 = Conv2D(filter,
                3,
                activation='relu',
                padding='same',
                kernel_initializer=f"{kernel_initializer}")(ublock9)
    
    conv10 = Conv2D(4, (1, 1), activation=activation_end, dtype='float32')(conv9)
    
    model = keras.Model(input, conv10, name='u-net')
    
    # Camada de saída
    model.compile(
        loss = getattr(keras.losses, str(loss))(),
        optimizer = getattr(keras.optimizers, str(optimizer))(learning_rate),
        metrics=['accuracy']
    )
    model.summary()

    return model