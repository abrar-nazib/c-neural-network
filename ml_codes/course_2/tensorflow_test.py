def sigmoid_vect(z: np.ndarray):
    """
    Computes the sigmoid of z on an numpy array.

    Args:
        z(ndarray): An array containing values

    Returns:
        g (scalar): An array containing sigmoid values
    """
    g = 1/(1+np.exp(-z))
    return
