import keras
from keras import layers
from keras.layers import Conv2D, Input, ReLU, BatchNormalization, Concatenate, Dropout, Conv2DTranspose, Concatenate, Conv2DTranspose
from keras_tuner import HyperParameters

def TrainModel(hp: HyperParameters): 
    # Camada de Entrada
    loss = hp.Choice('loss', ['BinaryCrossentropy'])
    optimizer = hp.Choice('optimizer', ['Adam'])
    # activation = hp.Choice("activation", ["relu"])
    activation_end = hp.Choice("activation_end", ["sigmoid"])
    pooling = hp.Choice('pooling', ['MaxPooling2D', 'AveragePooling2D'])
    # upscale = hp.Choice('upscale', ['Conv2DTranspose'])
    learning_rate = hp.Choice('learning_rate', values=[0.001])
    kernel_initializer = hp.Choice('kernel_initializer', ['he_normal'])
    hp_kernel_size = hp.Int('kernel_size', min_value=3, max_value=7, step=2)
    hp_dropout = hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
    
    
    def down_block(x, filters: int, use_maxpool=True):
        x = Conv2D(filters, hp_kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, hp_kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        if use_maxpool:
            return getattr(layers, str(pooling))((2, 2))(x), x
        return x
    
    def up_block(x, y, filters):
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = Concatenate(axis= 3)([x, y])
        x = Conv2D(filters, hp_kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, hp_kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    
    filter = [32, 64, 128, 256, 512]
    # encode
    input = Input(shape=(768, 512, 4))
    x, temp1 = down_block(input, filter[0]) # type: ignore
    x, temp2 = down_block(x, filter[1]) # type: ignore
    x, temp3 = down_block(x, filter[2]) # type: ignore
    x, temp4 = down_block(x, filter[3]) # type: ignore
    
    x = down_block(x, filter[4], use_maxpool = False)
    
    # decode 
    x = up_block(x, temp4, filter[3])
    x = up_block(x, temp3, filter[2])
    x = up_block(x, temp2, filter[1])
    x = up_block(x, temp1, filter[0])
    x = Dropout(hp_dropout)(x)
    
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
    # ID: 347 - val_accuracy: 0.87415 | 32 | 64 | 128 | 256 | 512 | 256 | 64 | 32 - 3 / 0.4
    loss = 'BinaryCrossentropy'
    optimizer = 'Adam'

    filter = [32, 64, 128, 256, 512]
    learning_rate = 0.001
    kernel_size = 3
    dropout = 0.2
    
    kernel_initializer = 'he_normal'
    activation_end = 'sigmoid'
    pooling = 'MaxPooling2D'

    def down_block(x, filters: int, use_maxpool=True):
        x = Conv2D(filters, kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        if use_maxpool:
            return getattr(layers, str(pooling))((2, 2))(x), x
        return x
    
    def up_block(x, y, filters):
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = Concatenate(axis= 3)([x, y])
        x = Conv2D(filters, kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel_size, kernel_initializer=f"{kernel_initializer}", padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    
    # encode
    input = Input(shape=(768, 512, 4))
    x, temp1 = down_block(input, filter[0]) # type: ignore
    x, temp2 = down_block(x, filter[1]) # type: ignore
    x, temp3 = down_block(x, filter[2]) # type: ignore
    x, temp4 = down_block(x, filter[3]) # type: ignore
    
    x = down_block(x, filter[4], use_maxpool = False)
    
    # decode 
    x = up_block(x, temp4, filter[3])
    x = up_block(x, temp3, filter[2])
    x = up_block(x, temp2, filter[1])
    x = up_block(x, temp1, filter[0])
    x = Dropout(dropout)(x)
    
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
