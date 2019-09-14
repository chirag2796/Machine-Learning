import tensorflow as tf

class EpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') >= 0.998:
            self.model.stop_training = True

def train_cifar100_conv():
    cifar100 = tf.keras.datasets.cifar100
    (training_images, training_labels), (test_images, test_labels) = cifar100.load_data()

    image_x_size = training_images[0].shape[0]
    image_y_size = training_images[0].shape[1]
    image_pixel_size = training_images[0].shape[2]

    training_images = training_images / 255.0
    training_images = training_images.reshape(len(training_images), image_x_size, image_y_size, image_pixel_size)
    test_images = test_images / 255.0
    test_images = test_images.reshape(len(test_images), image_x_size, image_y_size, image_pixel_size)

    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(image_x_size, image_y_size, image_pixel_size)),
                                        tf.keras.layers.MaxPool2D(2,2),
                                        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                        tf.keras.layers.MaxPool2D(2,2),
                                        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                                        tf.keras.layers.MaxPool2D(2, 2),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(units=100, activation=tf.nn.softmax)])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(training_images, training_labels, validation_data=(test_images, test_labels), epochs=30, callbacks=[EpochCallback()])
    return history

train_cifar100_conv()