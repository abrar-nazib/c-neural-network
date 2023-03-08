
import numpy as np
import math
import matplotlib.pyplot as plt

# autopep8: off
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow import keras
# autopep8: on


def sigmoid(z: float):
    """
    Computes the sigmoid of z on a float value

    Args:
        z(float): Sigmoid function for Single example

    Returns:
        g (scalar): An array containing sigmoid values
    """
    g = 1/(1+math.exp(-z))
    return g


def sigmoid_arr(z: np.ndarray):
    """
    Computes the sigmoid of z on an numpy array. Useful in applying sigmoid on multiple values at once

    Args:
        z(ndarray): An array containing values

    Returns:
        g (ndarray): An array containing sigmoid values
    """
    g = 1/(1+np.exp(-z))
    return g


def dense(A_in: np.ndarray, W: np.ndarray, B: np.ndarray):
    """
    Create a dense layer of neurons and compute the output of the layer    

    Args:
        A_in (np.ndarray (1, n)): Inputs from previous layer
        W (np.ndarray (n, j)): Weights for every neuron 
        B (np.ndarray (1, j)) : Biases for every neuron
    """
    units = W.shape[1]
    A_out = np.matmul(A_in, W) + B
    return A_out


def sequential(X_in, W1, B1, W2, B2):
    a1 = sigmoid_arr(dense(X_in, W1, B1))
    a2 = sigmoid(dense(a1, W2, B2))
    return 1 if a2 > 0.5 else 0


def normalize_data(X: np.ndarray):
    """
    Normalizes the training data

    Args:
        X (np.ndarray): Input Training matrix

    Returns:
        normalizer(callable): Normalizer object for training dataset
        Xn (np.ndarray): Normalized Training Data
    """
    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(X)  # Learns the mean and variance from the shape of X
    X_normalized = normalizer(X)
    return normalizer, X_normalized


def load_coffee_data():
    """ Creates a coffee roasting data set.
        roasting duration: 12-15 minutes is best
        temperature range: 175-260C is best
    """
    rng = np.random.default_rng(2)
    # print(rng)
    X = rng.random(400).reshape(-1, 2)
    # print(X)
    X[:, 1] = X[:, 1] * 4 + 11.5          # 12-15 min is best
    X[:, 0] = X[:, 0] * (285-150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))

    i = 0
    for t, d in X:
        y = -3/(260-175)*t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d <= y):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1, 1))


if __name__ == "__main__":
    W1_tmp = np.array([[-8.93,  0.29, 12.9],
                       [-0.1,  -7.32, 10.81]])
    b1_tmp = np.array([-9.82, -9.28,  0.96])
    W2_tmp = np.array([[-31.18], [-27.59], [-32.56]])
    b2_tmp = np.array([15.41])

    X_tst = np.array([
        [200, 13.9],  # postive example
        [200, 17]])   # negative example
    normalizer, X_tstn = normalize_data(X_tst)  # remember to normalize
    X_numpy = X_tstn.numpy()
    a_in = X_numpy[1, :].reshape(1, -1)
    print(sequential(a_in, W1_tmp, b1_tmp, W2_tmp, b2_tmp))
