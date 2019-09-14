import tensorflow as tf

class EpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') >= 0.998:
            self.model.stop_training = True

def train_mnist_conv():
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images / 255.0
    training_images = training_images.reshape(len(training_images), 28,28,1)
    test_images = test_images / 255.0
    test_images = test_images.reshape(len(test_images), 28, 28, 1)

    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
                                        tf.keras.layers.MaxPool2D(2,2),
                                        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                        tf.keras.layers.MaxPool2D(2,2),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(training_images, training_labels, validation_data=(test_images, test_labels), epochs=20, callbacks=[EpochCallback()])
    return history

train_mnist_conv()