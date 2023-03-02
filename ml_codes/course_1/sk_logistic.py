from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

X_test = X.copy()
y_test = y.copy()

lr_model = LogisticRegression()  # Creates a logistic regression Object
lr_model.fit(X, y)  # X being training feature matrix, y being output vector

y_pred = lr_model.predict(X_test)  # Testing the model on trained data
print(y_pred)

# Testing accuracy of the model
print(lr_model.score(X_test, y_test))
