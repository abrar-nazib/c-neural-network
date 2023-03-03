import numpy as np
import math


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
    Create a dense layer of neurons    
    """

    # units = W.shape[1]
    # a_out = np.zeros(units)
    # for j in range(units):
    #     z = np.dot(a_in, W[:, j]) + b[j]
    #     a_out[j] = sigmoid(z)
    # return a_out
    Z = np.matmul(A_in, W) + B
    A_out = sigmoid_arr(Z)
    return A_out


def sequential(x):
    # a1 = dense(x, W1, b1)
    # a2 = dense(a1, W2, b2)
    # a3 = dense(a2, W3, b3)
    # f_x = a3
    # return f_x
    return


W = np.array([
    [1, -3, 5],
    [2, 4, -6]
])
b = np.array([[-1, 1, 2]])
a_in = np.array([[-2, 4]])
print(dense(a_in, W, b))
