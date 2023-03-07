import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
import numpy as np

X = np.array([[1, 2, 3, 4]])
Y = np.array([[1, 2, 3, 4]])

model = Sequential([
    Dense(units=25, activation='sigmoid'),
    Dense(units=15, activation='sigmoid'),
    Dense(units=1, activation='sigmoid'),
])


# Specify the loss function
model.compile(loss=BinaryCrossentropy())

### Run Gradient Descent ###
# Here epochs is the number of iterations to perform
model.fit(X, Y, epochs=100)
