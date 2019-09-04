import pandas as pd
from tensorflow import keras

pad_sequences = keras.preprocessing.sequence.pad_sequences

df = pd.DataFrame([[1,2,1], [1]], columns=['A', 'B', 'C'])
print(df)
df = pad_sequences([[1,2,1], [1]], maxlen=5)
print(df)