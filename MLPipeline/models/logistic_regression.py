import copy

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from MLPipeline.transformation import transform_data, train_test_data


def gradient_descent(x, y_true, y_pred):
    difference = y_pred - y_true
    num_samples = y_true.shape[0]
    gradients_w = (1 / num_samples) * (x.T @ difference)
    return gradients_w


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.array([np.sum(e, axis=1)]).T


class MultinomialLogisticRegression:

    def predict_prob(self, x):
        return softmax(np.dot(x, self.weights))

    def predict(self, x):
        return np.argmax(self.predict_prob(x), axis=1)

    def predict_score(self, x, y):
        xb = np.hstack((x, np.ones((x.shape[0], 1))))  # shape = (num_samples, features+1(bias))
        return accuracy_score(y, self.predict(xb))

    def fit(self, x, y, iterations=200, learning_rate=0.01):
        x = copy.deepcopy(x)  # shape = (samples, features)
        y = copy.deepcopy(y)
        bias = np.ones((x.shape[0], 1))
        xb = np.hstack((x, bias))  # shape = (num_samples, features+1(bias))

        encoder = OneHotEncoder(sparse=False)
        y_encoded = encoder.fit_transform(y)
        n_classes = y_encoded.shape[1]
        n_features = xb.shape[1]  # includes bias col

        # Init default params
        self.weights = np.zeros((n_features, n_classes))  # shape = (num_samples, features+1(bias))

        for _ in tqdm(range(iterations)):
            predictions = self.predict_prob(xb)
            error_w = gradient_descent(xb, y_encoded, predictions)
            self.weights -= learning_rate * error_w


if __name__ == '__main__':
    data_path = "/home/j3/Desktop/gesture-recognition/data/clean/gestures.pkl"
    df = pd.read_pickle(data_path)
    df_ready = transform_data(df)
    X_train, X_test, y_train, y_test = train_test_data(df_ready)
    # model = LogisticRegression()
    # model.fit(X_train, y_train, iterations=100)
    # print(model.score)
    # # print(model.score())

    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=8,
                               random_state=1)

    model = MultinomialLogisticRegression()
    model.fit(X, y.reshape(-1, 1), iterations=200)
    # model.fit(X_train, y_train, iterations=100)

    accuracy = model.predict_score(X, y)
    # accuracy = model.predict_score(X_train, y_train)
    print(accuracy * 100)
