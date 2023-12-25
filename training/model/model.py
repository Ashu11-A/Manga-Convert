import keras
from keras import layers

def LoaderModel():
    model = keras.Sequential([
        layers.Input(shape=(768, 512, 4)),
        #layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.4),
        # layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        # layers.Dropout(0.2),
        layers.Dense(4, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.001, amsgrad=True),
        metrics=['accuracy'],
        run_eagerly=True
    )

    model.summary()

    return model
