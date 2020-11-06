import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tensorflow as tf

dataset = [
    [1,2,3,4],
    [1,2,3,40],
    [1,3,3,4]
]

df = pd.DataFrame(dataset)
X = df.values
norm_x = tf.keras.utils.normalize(X)
print(norm_x)