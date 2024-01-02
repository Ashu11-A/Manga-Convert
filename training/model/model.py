import keras
from keras import layers
from keras_tuner import HyperParameters
import tensorflow as tf

def TrainModel(hp: HyperParameters): 
    # Camada de Entrada
    loss = hp.Choice('loss', ['BinaryCrossentropy'])
    optimizer = hp.Choice('optimizer', ['Adam'])
    activation = hp.Choice("activation", ["relu"])
    activation_end = hp.Choice("activation_end", ["sigmoid"])
    pooling = hp.Choice('pooling', ['MaxPooling2D'])
    learning_rate = hp.Choice('learning_rate', values=[0.001])

    hp_filters_1 = hp.Int('filters_1', min_value=16, max_value=32 , step=16)
    hp_filters_2 = hp.Int('filters_2', min_value=32, max_value=64 , step=32)
    hp_filters_3 = hp.Int('filters_3', min_value=64, max_value=128 , step=64)
    hp_filters_4 = hp.Int('filters_4', min_value=128, max_value=256 , step=128)
    hp_filters_5 = hp.Int('filters_5', min_value=256, max_value=512 , step=256)
    hp_kernel_size = hp.Int('kernel_size', min_value=3, max_value=7, step=2)
    
    # hp_regularizer = hp.Choice("regularizer", ['l1', 'l2'])
    # hp_momentum = hp.Float("momentum", 0.9, 0.99, step=0.03)
    hp_dropout = hp.Float('dropout_rate', 0.2, 0.5, step=0.1)
    
    inputs = layers.Input(shape=(768, 512, 4))
    
    c1 = layers.Conv2D(hp_filters_1, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv1_1')(inputs)
    c1 = layers.Dropout(hp_dropout)(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(hp_filters_1, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv1_2')(c1)
    p1 = getattr(layers, pooling)((2, 2))(c1) # type: ignore

    c2 = layers.Conv2D(hp_filters_2, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv2_1')(p1)
    c2 = layers.Dropout(hp_dropout)(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(hp_filters_2, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv2_2')(c2)
    p2 = getattr(layers, pooling)((2, 2))(c2) # type: ignore
    
    c3 = layers.Conv2D(hp_filters_3, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv3_1')(p2)
    c3 = layers.Dropout(hp_dropout)(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(hp_filters_3, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv3_2')(c3)
    p3 = getattr(layers, pooling)((2, 2))(c3) # type: ignore
    
    c4 = layers.Conv2D(hp_filters_4, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv4_1')(p3)
    c4 = layers.Dropout(hp_dropout)(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(hp_filters_4, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv4_2')(c4)
    p4 = getattr(layers, pooling)((2, 2))(c4) # type: ignore
    
    c5 = layers.Conv2D(hp_filters_5, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv5_1')(p4)
    c5 = layers.Dropout(hp_dropout)(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(hp_filters_5, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv5_2')(c5)

    #Expansive path 
    u6 = layers.Conv2DTranspose(hp_filters_4, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(hp_filters_4, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv6_1')(u6)
    c6 = layers.Dropout(hp_dropout)(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(hp_filters_4, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv6_2')(c6)
    
    u7 = layers.Conv2DTranspose(hp_filters_3, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(hp_filters_3, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv7_1')(u7)
    c7 = layers.Dropout(hp_dropout)(c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(hp_filters_3, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv7_2')(c7)
    
    u8 = layers.Conv2DTranspose(hp_filters_2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(hp_filters_2, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv8_1')(u8)
    c8 = layers.Dropout(hp_dropout)(c8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv2D(hp_filters_2, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv8_2')(c8)
    
    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(hp_filters_1, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv9_1')(u9)
    c9 = layers.Dropout(hp_dropout)(c9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv2D(hp_filters_1, hp_kernel_size, activation=activation, kernel_initializer='he_normal', padding='same', name='conv9_2')(c9)
    
    outputs = layers.Conv2D(4, (1, 1), activation=activation_end, dtype='float32')(c9)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    
    # Camada de saída
    model.compile(
        loss = getattr(keras.losses, loss)(), # type: ignore
        optimizer = getattr(keras.optimizers, optimizer)(learning_rate), # type: ignore
        metrics=['accuracy']
    )
    model.summary()

    return model

def LoaderModel():
    inputs = layers.Input(shape=(768, 512, 4))
    
    c1 = layers.Conv2D(8, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv1_1')(inputs)
    c1 = layers.Dropout(0.4)(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(8, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv1_2')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1) # type: ignore

    c2 = layers.Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv2_1')(p1)
    c2 = layers.Dropout(0.4)(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv2_2')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv3_1')(p2)
    c3 = layers.Dropout(0.4)(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv3_2')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv4_1')(p3)
    c4 = layers.Dropout(0.4)(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv4_2')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    c5 = layers.Conv2D(384, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv5_1')(p4)
    c5 = layers.Dropout(0.4)(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(384, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv5_2')(c5)

    #Expansive path 
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv6_1')(u6)
    c6 = layers.Dropout(0.4)(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv6_2')(c6)
    
    u7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv7_1')(u7)
    c7 = layers.Dropout(0.4)(c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv7_2')(c7)
    
    u8 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv8_1')(u8)
    c8 = layers.Dropout(0.4)(c8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv8_2')(c8)
    
    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(8, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv9_1')(u9)
    c9 = layers.Dropout(0.4)(c9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv2D(8, 3, activation='relu', kernel_initializer='he_normal', padding='same', name='conv9_2')(c9)

    outputs = layers.Conv2D(4, (1, 1), activation='sigmoid', dtype='float32')(c9)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    
    # Camada de saída
    model.compile(
        loss=keras.losses.BinaryCrossentropy(), # type: ignore
        optimizer = keras.optimizers.Adam(0.001), # type: ignore
        metrics=['accuracy']
    )
    model.summary()

    return model