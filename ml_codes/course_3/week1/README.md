# Highlight

The best Thing I implemented in this lab is using boolean algebra

```python
# UNQ_C2
# GRADED FUNCTION: select_threshold

def select_threshold(y_val, p_val):
    """
    Finds the best threshold to use for selecting outliers
    based on the results from a validation set (p_val)
    and the ground truth (y_val)

    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set

    Returns:
        epsilon (float): Threshold chosen
        F1 (float):      F1 score by choosing epsilon as threshold
    """

    best_epsilon = 0
    best_F1 = 0
    F1 = 0

    step_size = (max(p_val) - min(p_val)) / 1000

    for epsilon in np.arange(min(p_val), max(p_val), step_size):

        ### START CODE HERE ###

        # Behold! my boolean algebra
        preds = p_val < epsilon
        tp = preds & y_val
        fp = preds & (~y_val)
        fn = (~preds) & y_val
        # yoh!

        prec = np.sum(tp)/(np.sum(tp) + np.sum(fp))
        rec = np.sum(tp)/(np.sum(tp) + np.sum(fn))
        F1 = (2 * prec * rec)/(prec + rec)



        ### END CODE HERE ###

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1
```
