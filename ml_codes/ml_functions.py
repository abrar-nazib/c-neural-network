"""
This file stores the raw implementation of all ML functions.
Author: Nazib Abrar
"""

import numpy as np
import copy
import math


def sigmoid_vect(z: np.ndarray):
    """
    Computes the sigmoid of z on an numpy array.

    Args:
        z(ndarray): An array containing values

    Returns:
        g (scalar): An array containing sigmoid values
    """
    g = 1/(1+np.exp(-z))
    return g


def compute_cost_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    """
    Computes cost for a logistic regression using logistic cost function

    Args:
        X (ndarray (m, n)): Training input of m rows and n features
        y (ndarray (n, )): Training output array
        w (ndarray (n, )): Model Parameter, weights
        b (scalar): Model Parameter, bias

    Returns:
        cost (scalar): Cost of the model
    """

    m = X.shape[0]  # training examples
    n = X.shape[1]  # features
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid_vect(z_i)
        cost += (- y[i]*np.log(f_wb_i) - (1 - y[i])*(np.log(1 - f_wb_i)))
    cost = cost/m
    return cost


def test_compute_cost_logistic():
    X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [
                       3, 0.5], [2, 2], [1, 2.5]])  # (m,n)
    y_train = np.array([0, 0, 0, 1, 1, 1])

    w_tmp = np.array([1, 1])
    b_tmp = -3
    print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))


def compute_gradient_linear(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    """
    Computes the gradient for linear regression. Gradient is the partial derivative term
    Args:
        X (ndarray (m, n)): Training input matrix
        y (ndarray (m, )): Training output vector
        w (ndarray (n, )): Model parameter, weight vector
        b (scalar): Model parameter, bias
    Returns:
        dj_dw (ndarray (n, )): Vector of derivative terms or gradient of weights
        dj_db (scalar): Gradient term of bias
    """

    m, n = X.shape
    dj_db = 0
    dj_dw = np.zeros(shape=(n, ))
    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return (dj_dw, dj_db)


def compute_gradient_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    """
    Computes the gradient terms for a linear regression function

    Args:
        X (ndarray (m, n)): Training input of n features containing m examples
        y (n
        darray (n, )): Training output array
        w (ndarray (n, )): Model parameter, weights
        b (scalar): Model parameter, bias

    Returns:
        dj_dw (ndarray, (n, )): Array containing gradient terms of weights
        dj_db (scalar): Gradient term of bias
    """
    m, n = X.shape
    dj_dw = np.zeros(shape=(n,))
    dj_db = 0.0
    for i in range(m):
        f_wb_i = sigmoid_vect(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]   # err_i is scalar
        for j in range(n):
            dj_dw[j] += err_i * X[i, j]
        dj_db += err_i

    return dj_dw/m, dj_db/m


def gradient_descent_logistic(X, y, w_in, b_in, alpha, num_iters):
    """
    Performs batch gradient descent

    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter
    """
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
        w = w - dj_dw
        b = b - dj_db

        J_history.append(compute_cost_logistic(X, y, w, b))
        if (i % math.ceil(num_iters/10) == 0):
            print(f"Iteration: {i}, Cost: {J_history[-1]}")
    return w, b, J_history


def compute_cost_linear_regularized(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 1.0):
    """
    Computes cost for a linear regression model with regularization

    Args:
        X (ndarray (m, n)): Input training features
        y (ndarray (m, )): Training outputs
        w (ndarray (n, )): Model parameters. Weights
        b (scalar): Model parameter. Bias
        lambda_ (scalar): Regularization parameter

    Returns:
        total_cost (scalar): Cost
    """

    m, n = X.shape
    cost = 0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        sq_err = (f_wb_i - y[i]) ** 2
        cost += sq_err
    cost = cost/(2*m)

    w_updated = (lambda_/(2*m))*(w**2)
    reg_cost = np.sum(w_updated)

    total_cost = cost + reg_cost
    return total_cost


def test_regularized_cost_linear():
    np.random.seed(1)
    X_tmp = np.random.rand(5, 6)
    y_tmp = np.array([0, 1, 0, 1, 0])
    w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
    b_tmp = 0.5
    lambda_tmp = 0.7
    cost_tmp = compute_cost_linear_regularized(
        X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

    print("Regularized cost:", cost_tmp)


def compute_cost_logistic_regularized(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 1):
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """
    m, n = X.shape
    cost = compute_cost_logistic(X, y, w, b)
    w_updated = (lambda_/(2*m))*(w**2)
    reg_cost = np.sum(w_updated)
    total_cost = cost+reg_cost
    return total_cost


def test_regularized_cost_logistic():
    np.random.seed(1)
    X_tmp = np.random.rand(5, 6)
    y_tmp = np.array([0, 1, 0, 1, 0])
    w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
    b_tmp = 0.5
    lambda_tmp = 0.7
    cost_tmp = compute_cost_logistic_regularized(
        X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

    print("Regularized cost:", cost_tmp)


def compute_gradient_linear_regularized(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 1):
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape  # (number of examples, number of features)
    dj_dw, dj_db = compute_gradient_linear(X, y, w, b)
    w_reg = (w * lambda_)/m
    dj_dw += w_reg

    return dj_dw, dj_db


def test_gradient_linear_regularized():
    np.random.seed(1)
    X_tmp = np.random.rand(5, 3)
    y_tmp = np.array([0, 1, 0, 1, 0])
    w_tmp = np.random.rand(X_tmp.shape[1])
    b_tmp = 0.5
    lambda_tmp = 0.7
    dj_dw_tmp, dj_db_tmp = compute_gradient_linear_regularized(
        X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

    print(f"dj_db: {dj_db_tmp}", )
    print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )


def compute_gradient_logistic_reg(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 1.0):
    """
    Computes the gradient for logistic regression 

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
    w_reg = (w * lambda_)/m
    dj_dw += w_reg

    return dj_dw, dj_db


def test_gradient_logistic_regularized():
    np.random.seed(1)
    X_tmp = np.random.rand(5, 3)
    y_tmp = np.array([0, 1, 0, 1, 0])
    w_tmp = np.random.rand(X_tmp.shape[1])
    b_tmp = 0.5
    lambda_tmp = 0.7
    dj_dw_tmp, dj_db_tmp = compute_gradient_logistic_reg(
        X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

    print(f"dj_db: {dj_db_tmp}", )
    print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )


if __name__ == "__main__":
    # test_compute_cost_logistic() # Testing logistic cost function
    # Testing regularized cost function for linear regression
    # test_regularized_cost_linear()
    # test_regularized_cost_logistic()
    # test_gradient_linear_regularized()
    test_gradient_logistic_regularized()
    pass
