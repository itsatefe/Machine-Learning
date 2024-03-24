import numpy as np

class LinearRegression:
    def __init__(self, dim, lr = 0.01, iterations = 10):
        self.lr = lr
        self.iterations = iterations
        self.params = np.zeros((dim, 1))

    def _gradient_descent(self, data, labels):
        y_hat = data @ self.params
        grads = data.T @ (y_hat - labels) / len(data)
        new_params = self.params - self.lr * grads
        return new_params

    def fit(self, data, labels):
        for i in range(self.iterations):
            self.params = self._gradient_descent(data, labels)

    def predict(self, data):
        return data @ self.params