import keras
from keras import layers

def LoaderModel():
    model = keras.Sequential (
        [
            keras.Input(shape=[768, 512, 4]),
            layers.Dense(units=128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(units=64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(units=4, activation='sigmoid'),
        ])

    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.SGD(),
        metrics=keras.metrics.Accuracy(),
        # run_eagerly=True
    )

    model.summary()

    return model