# https://www.youtube.com/watch?v=QiLHwCkx-YQ&feature=emb_logo

import warnings
warnings.simplefilter(action="ignore")

import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import keras

import numpy as np

from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras import regularizers

import pandas as pd

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import matplotlib
from matplotlib import pyplot as plt

DATASET_FOLDERPATH = r"C:\Projects\DEV\Datasets\Classification\Breast_Cancer\\"
df1 = pd.read_csv(DATASET_FOLDERPATH + r"X_data.csv")
df2 = pd.read_csv(DATASET_FOLDERPATH + r"Y_data.csv")

df1 = preprocessing.scale(df1)

X_train, X_test, y_train, y_test = train_test_split(df1, df2, test_size=0.2)

# 'shallow' Logistic Regression model
model = Sequential()
model.add(Dense(13, input_shape=(30,), activation='relu')) #here n(13) is arbitrary as neurons will be of input shape i.e. 30
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

history = model.fit(X_train, y_train, epochs=2000, validation_split=0.15, verbose=2, callbacks=[earlystopper])
history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
plt.figure()
plt.figure()
plt.plot(loss_values, 'b', label='training_loss')
plt.plot(val_loss_values, 'r', label='val trainaing loss')
plt.legend()
plt.xlabel("Epochs")
plt.show()

accuracy_values = history_dict['acc']
val_accuracy_values = history_dict['val_acc']
plt.plot(val_accuracy_values, '-g', label='val_acc')
plt.plot(accuracy_values, '-r', label='acc')
plt.legend()
plt.show()

# Calculate loss and accuracy of testing data
loss, acc = model.evaluate(X_test, y_test)
print("Test loss: ", loss)
print("Test accuracy: ", acc)

# AUC Scoring of testing data
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
y_test_pred = model.predict_proba(X_test)
fpr_keras, tpr_keras, threshold_keras = roc_curve(y_test, y_test_pred)
auc_keras = auc(fpr_keras, tpr_keras)
print('Testing data AUC: ', auc_keras)

# ROC Curve of testing data
plt.figure(1)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False Positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()