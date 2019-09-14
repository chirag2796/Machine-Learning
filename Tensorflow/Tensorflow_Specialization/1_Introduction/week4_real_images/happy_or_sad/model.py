import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

train_dataset_filepath = r"D:\Dev\Datasets\Images\happy-or-sad"
train_happy_dir = train_dataset_filepath + r"\happy"
train_sad_dir = train_dataset_filepath + r"\sad"

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(train_dataset_filepath,
                                                        target_size=(150, 150), batch_size=10,
                                                        class_mode="binary")

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=0.001), metrics=['acc'])
model.fit_generator(train_generator, epochs=20, steps_per_epoch=4)