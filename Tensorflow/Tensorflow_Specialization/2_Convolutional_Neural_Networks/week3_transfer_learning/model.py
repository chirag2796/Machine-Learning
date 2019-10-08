import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import os, time

NAME = "Cats_vs_dogs_{}".format(int(time.time()))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=r"files\logs\no_dropout_{}".format(NAME), write_graph=True, write_grads=True,
                                             write_images=True, histogram_freq=0, update_freq='epoch')

base_dir = r"D:\Dev\Datasets\Images\cats_and_dogs_filtered"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.,
                                                                rotation_range=40,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                shear_range=0.2,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True,
                                                                fill_mode='nearest')
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.)

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=20, class_mode='binary',
                                                    target_size=(150, 150))
validation_generator = test_datagen.flow_from_directory(validation_dir, batch_size=20, class_mode='binary',
                                                        target_size=(150, 150))
##################################################################################

local_weights_file = r"D:\Dev\ML\Trained_Models\Image_Classification\Inception\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

pre_trained_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=(150, 150, 3),
                                                                   include_top=False,
                                                                   weights=None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False
# print(pre_trained_model.summary())

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = tf.keras.layers.Flatten()(last_output)
# Add a fully connected layer with 1024 hidden units and relu activation
x = tf.keras.layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = tf.keras.layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(pre_trained_model.input, x)
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])

history = model.fit_generator(train_generator, validation_data=validation_generator, steps_per_epoch=100,
                              epochs=20, validation_steps=50, callbacks=[tensorboard])

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()
