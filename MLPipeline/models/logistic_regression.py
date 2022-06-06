import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


class MultinomialLogisticRegression:

    def gradient_descent(x, y_true, y_pred):
        difference = y_pred - y_true
        gradient_b = np.mean(difference, axis=0)
        gradients_w = np.mean(np.matmul(x.transpose(), difference))
        return gradients_w, gradient_b

    def predict_prob(self, x):
        return softmax(np.dot(x, self.weights) + self.bias, axis=1)

    def predict(self, x):
        return np.argmax(self.predict_prob(x), axis=1)

    def predict_score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def fit(self, x, y, iterations=200, learning_rate=0.01):
        encoder = OneHotEncoder(sparse=False)
        y_encoded = encoder.fit_transform(y)
        n_classes = y_encoded.shape[1]
        n_features = x.shape[1]

        # Init default params
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        for _ in range(iterations):
            predictions = self.predict_prob(x)
            error_w, error_b = MultinomialLogisticRegression.gradient_descent(x, y_encoded, predictions)

            # Update model params
            self.weights -= learning_rate * error_w
            self.bias -= learning_rate * error_b
            # learning_rate *= 0.95
