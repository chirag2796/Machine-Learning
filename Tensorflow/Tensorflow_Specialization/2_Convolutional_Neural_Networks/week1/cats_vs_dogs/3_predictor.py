import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model(r"files\cats_vs_dogs.h5")
image_path = r"D:\Dev\Datasets\Images\cats_and_dogs_filtered\predictions\dog1.jpeg"
img = tf.keras.preprocessing.image.load_img(path=image_path, target_size=(150,150,3))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0] > 0:
    print("Dog")
else:
    print("Cat")