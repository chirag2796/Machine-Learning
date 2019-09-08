import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tensorflow as tf

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)])

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(x=training_images, y=training_labels, validation_data=(test_images, test_labels), epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
model.save("files\\fashion_cnn.h5")
