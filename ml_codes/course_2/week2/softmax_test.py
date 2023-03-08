import numpy as np
import matplotlib.pyplot as plt

#fmt: off
import os
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


import tensorflow as tf

### Uncomment tensorflow.keras before model testing ###
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LeakyReLU
# from tensorflow.keras.activations import linear, relu, sigmoid
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import SparseCategoricalCrossentropy

### Comment keras before model training ###
from keras import Sequential
from keras.layers import Dense, LeakyReLU
from keras.activations import linear, relu, sigmoid
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

from sklearn.datasets import make_blobs

#fmt: on


def softmax(z: np.ndarray):
    """Calculates the output array using softmax algorithm on a given array z

    Args:
        z (np.ndarray (n, )): Vector/array containing n values 
    """
    ez = np.exp(z)  # Do exponential operation in all elements
    a_out = ez/np.sum(ez)  # Each element of the array will be (ez_i)/(sum(ez))
    return a_out  # each element of a_out will range between 0-1


def train_model(X: np.ndarray, y: np.ndarray, epochs: int):
    model = Sequential([
        Dense(units=25, activation="relu"),
        Dense(units=15, activation="relu"),
        # Softmax activation function usage
        Dense(units=4, activation="softmax"),
    ])

    model.compile(optimizer=Adam(0.001), loss=SparseCategoricalCrossentropy())
    model.fit(X, y, epochs=epochs)
    return model


def train_model_preffered(X: np.ndarray, y: np.ndarray, epochs: int):
    model = Sequential([
        Dense(units=25, activation="relu"),
        Dense(units=15, activation="relu"),
        # Softmax activation function usage
        Dense(units=4, activation="linear"),
    ])

    model.compile(optimizer=Adam(0.001),
                  loss=SparseCategoricalCrossentropy(from_logits=True))
    model.fit(X, y, epochs=epochs)
    return model


if __name__ == "__main__":
    # Generating Dataset
    centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
    X_train, y_train = make_blobs(
        n_samples=2000, centers=centers, cluster_std=1.0, random_state=30)
    model = train_model_preffered(X=X_train, y=y_train, epochs=30)
    prediction = model.predict(X_train)
    prediction = tf.nn.softmax(prediction).numpy()
    print(
        f"Highest Value of Prediction: {np.max(prediction)}\nLowest value of prediction: {np.min(prediction)}")
    for elem in prediction:
        print(np.argmax(elem))
