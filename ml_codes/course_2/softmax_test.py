from keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras import Sequential
from keras.losses import SparseCategoricalCrossentropy
X = [[10]]
Y = [[10]]

model = Sequential([
    Dense(units=25, activation="relu"),
    Dense(units=15, activation="relu"),
    Dense(units=10, activation="softmax"),
])

model.compile(loss=SparseCategoricalCrossentropy())

model.fit()

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss=SparseCategoricalCrossentropy(from_logits=True))
model.fit(X, Y, epochs=10)
