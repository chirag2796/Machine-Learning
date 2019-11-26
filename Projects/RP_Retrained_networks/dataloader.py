import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf


class DataLoader:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    @staticmethod
    def get_training_data(number_of_images):
        x = DataLoader.x_train[: number_of_images]
        y = DataLoader.y_train[: number_of_images]
        return x, y