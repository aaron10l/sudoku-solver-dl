from functools import reduce

import numpy as np
import tensorflow as tf

SUB_SAMPLE = 10000
TEST = 100

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.list_logical_devices('GPU')
  except RuntimeError as e:
    print(e)

def get_test_train():
    def to_board(string):
        return np.array(list(string), dtype=int).reshape((9, 9))

    def to_features_labels(puzzles):
        features = puzzles[:,0].reshape((-1, 9, 9, 1))
        labels = np.eye(9)[puzzles[:,1] - 1]
        return features, labels

    puzzles = np.loadtxt("../Data/sudoku.csv", delimiter=",", dtype=str)[1:SUB_SAMPLE + 1]
    puzzles = np.vectorize(to_board, signature="()->(9,9)")(puzzles)

    rng = np.random.default_rng(seed=42)
    rng.shuffle(puzzles)

    split = SUB_SAMPLE - TEST
    return to_features_labels(puzzles[split:]), to_features_labels(puzzles[:split])

def train_model(model, name):
    test, train = get_test_train()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
        ],
    )

    model.fit(
        x=train[0],
        y=train[1],
        validation_data=(test[0], test[1]),
        batch_size=1,
        epochs=100,
        callbacks=tf.keras.callbacks.EarlyStopping(monitor="val_categorical_accuracy", patience=1)
    )

    model.save(name + ".h5")

    # predictions = model.predict(
    #     x=test[0],
    #     batch_size=1
    # )
    # print(np.reshape(test[0][0], (9,9)))
    # print(np.argmax(predictions[0], axis=2) + 1)
    # print(np.argmax(test[1][0], axis=2) + 1)

stanford = tf.keras.Sequential([
    tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(9, activation="softmax")
])

basic_cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(9, activation="softmax")
])

def iterative_model():
    # This helps jump start the models but doesn't improve end performance
    class CombineGiven(tf.keras.Layer):
        def call(self, x, given):
            given_one_hot = tf.one_hot(tf.cast(tf.reshape(given, [-1,9,9]) - 1, dtype=tf.int32), 9)
            return given_one_hot + x 

    x_input = tf.keras.layers.Input((9, 9, 1))
    x = x_input

    x = tf.keras.layers.Conv2D(512, 9, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(512, 9, padding="same", activation="relu")(x)
    x = tf.keras.layers.Dense(9, activation="softmax")(x)
    x = CombineGiven()(x, x_input)

    x = tf.keras.layers.Conv2D(512, 9, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(512, 9, padding="same", activation="relu")(x)
    x = tf.keras.layers.Dense(9, activation="softmax")(x)
    x = CombineGiven()(x, x_input)

    x = tf.keras.layers.Conv2D(512, 9, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(512, 9, padding="same", activation="relu")(x)
    x = tf.keras.layers.Dense(9, activation="softmax")(x)
    x = CombineGiven()(x, x_input)

    return tf.keras.Model(inputs=x_input, outputs=x)

train_model(iterative_model(), "resnet")
