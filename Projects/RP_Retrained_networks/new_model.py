import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
from Projects.RP_Retrained_networks.dataloader import DataLoader

NAME = "mnist_{}".format(int(time.time()))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=r"files\logs\{}".format(NAME))

mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()
number_of_images = 25

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

def get_new_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.add(tf.keras.layers.Dense(8, activation=tf.nn.softmax))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

pass_input_sizes = [25, 30, 40, 50, 60, 70, 80, 100, 120]

pass_x_train = x_train[:pass_input_sizes[0]]
pass_y_train = y_train[:pass_input_sizes[0]]
model = get_new_model()
model.fit(pass_x_train, pass_y_train, epochs=40, callbacks=[tensorboard], validation_data=(x_test, y_test), verbose=2)

for i in range(1, len(pass_input_sizes)):
    del model
    model = get_new_model()
    number_of_images = pass_input_sizes[i]
    pass_x_train = x_train[:number_of_images]

    next_pass_x = x_train[pass_input_sizes[0]: pass_input_sizes[i]]
    predictions = [np.argmax(pred) for pred in model.predict(next_pass_x)]
    pass_y_train = np.concatenate((y_train[:pass_input_sizes[0]], predictions), axis=0)
    model.fit(pass_x_train, pass_y_train, epochs=32, callbacks=[tensorboard], validation_data=(x_test, y_test), verbose=2)


