
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
        f_wb_i = sigmoid_arr(z_i)
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


if __name__ == "__main__":
    test_compute_cost_logistic()
