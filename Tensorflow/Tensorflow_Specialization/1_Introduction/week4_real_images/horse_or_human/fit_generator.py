import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

train_dataset_filepath = r"D:\Dev\Datasets\Images\horses-or-humans\train"
test_dataset_filepath = r"D:\Dev\Datasets\Images\horses-or-humans\validation"
train_horse_dir = train_dataset_filepath + r"\horses"
train_human_dir = train_dataset_filepath + r"\humans"

def previsualization():
    train_horse_names = os.listdir(train_horse_dir)
    # print(train_horse_names[:10])
    train_human_names = os.listdir(train_human_dir)
    # print(train_human_names[:10])

    nrows = 4
    ncols = 4
    pic_index = 0

    fig = plt.gcf()
    fig.set_size_inches(ncols*4, nrows*4)
    pic_index += 8
    next_horse_pix = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[pic_index-8:pic_index]]
    next_human_pix = [os.path.join(train_human_dir, fname) for fname in train_human_names[pic_index-8:pic_index]]

    for i, img_path in enumerate(next_horse_pix + next_human_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()

# All images will be rescaled bu 1./255
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dataset_filepath,
                                                    target_size=(300, 300), batch_size=128,
                                                    class_mode="binary") #bcz we use binary_crosentropy loss
validation_generator = validation_datagen.flow_from_directory(test_dataset_filepath,
                                                    target_size=(300, 300), batch_size=32,
                                                    class_mode="binary")


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPool2D(2,2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.RMSprop(lr=0.001), metrics=['acc'])

history = model.fit_generator(train_generator, steps_per_epoch=8, epochs=15,
                              validation_data=validation_generator,
                              validation_steps=8)

model.save("files\\horse_or_human.h5")