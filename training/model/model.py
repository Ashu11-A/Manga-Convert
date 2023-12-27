import keras
from keras import layers
from keras_tuner import HyperParameters

def TrainModel(hp: HyperParameters):
    model = keras.Sequential()
    
    model.add(layers.Input(shape=(768, 512, 4)))
    
    hp_filters_1 = hp.Int('filters1', min_value=4, max_value=64, step=2)
    model.add(layers.Conv2D(filters=hp_filters_1, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    hp_upscale_1 = hp.Int('upscale1', min_value=32, max_value=64, step=2)
    model.add(layers.Conv2DTranspose(filters=hp_upscale_1, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    
    hp_filters_2 = hp.Int('filters2', min_value=4, max_value=64, step=2)
    model.add(layers.Conv2D(filters=hp_filters_2, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    hp_upscale_2 = hp.Int('upscale2', min_value=32, max_value=64, step=2)
    model.add(layers.Conv2DTranspose(filters=hp_upscale_2, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    
    hp_units_1 = hp.Int('units1', min_value=8, max_value=256, step=2)
    model.add(layers.Dense(units=hp_units_1, activation='relu'))

    hp_units_2 = hp.Int('units2', min_value=8, max_value=256, step=2)
    model.add(layers.Dense(units=hp_units_2, activation='relu'))

    hp_units_3 = hp.Int('units3', min_value=8, max_value=256, step=2)
    model.add(layers.Dense(units=hp_units_3, activation='relu'))
    
    # hp_units_end = hp.Int(name='units', min_value=4, max_value=10, step=32)
    model.add(layers.Dense(units=4, activation='sigmoid'))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001])

    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), # type: ignore
        metrics=['accuracy'],
        # run_eagerly=True
    )

    model.summary()

    return model

def LoaderModel():
    model = keras.Sequential()
    
    model.add(layers.Input(shape=(768, 512, 4)))
    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.Dense(units=16, activation='relu'))
    model.add(layers.Dense(units=4, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        metrics=['accuracy'],
    )

    return model

