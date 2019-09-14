import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("files\\horse_or_human.h5")
image_path = r"D:\Dev\Datasets\Images\horses-or-humans\predictions\horsehuman1.jpg"
img = tf.keras.preprocessing.image.load_img(path=image_path, target_size=(300,300,3))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0] > 0.5:
    print("Human")
else:
    print("Horse")