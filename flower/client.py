import flwr as fl
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, y_train = x_train[int(0.15*len(x_train)):-1], y_train[int(0.15*len(x_train)):-1]

#model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
use_nchw_format = False
data_format = 'channels_first' if use_nchw_format else 'channels_last'
data_shape = (1, 28, 28) if use_nchw_format else (28, 28, 1)
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=data_shape, data_format=data_format))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', data_format=data_format))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format=data_format))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=10, batch_size=20, steps_per_epoch=3)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}
    

fl.client.start_numpy_client("[::]:8080", client=CifarClient())