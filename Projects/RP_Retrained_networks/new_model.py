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

