import copy

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from MLPipeline.transformation import transform_data, train_test_data


def gradient_descent(x, y_true, y_pred):
    difference = y_pred - y_true
    num_samples = y_true.shape[0]
    gradients_w = (1 / num_samples) * (x.T @ difference)
    return gradients_w


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class MultinomialLogisticRegression:

    @staticmethod
    def loss(X, Y, W):
        Z = X @ W
        N = X.shape[0]
        loss = 1 / N * (-1 * np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
        return loss

    def predict_prob(self, x):
        return softmax(np.dot(x, self.weights))

    def predict(self, x):
        return np.argmax(self.predict_prob(x), axis=1)

    def predict_score(self, x, y):
        xb = np.hstack((x, np.ones((x.shape[0], 1))))  # shape = (num_samples, features+1(bias))
        predictions = self.predict_prob(xb)

        encoder = OneHotEncoder(sparse=False)
        y_encoded = encoder.fit_transform(y)

        score = 0
        for i, j in zip(predictions, y_encoded):
            if np.argmax(i) == np.argmax(j):
                score += 1

        return score / xb.shape[0]

    def fit(self, x, y, iterations=1000, learning_rate=0.1, save_loss=False):
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
        self.loss = []

        for _ in tqdm(range(iterations)):
            predictions = self.predict_prob(xb)
            error_w = gradient_descent(xb, y_encoded, predictions)
            self.weights -= learning_rate * error_w

            # Loss
            loss = MultinomialLogisticRegression.loss(xb, y_encoded, self.weights)
            self.loss.append(loss)

        if save_loss:
            loss_df = pd.DataFrame(self.loss, columns=["loss"])
            loss_df.to_csv("loss.csv", index=False)


if __name__ == '__main__':
    data_path = "/home/j3/Desktop/gesture-recognition/data/clean/gestures.pkl"
    df = pd.read_pickle(data_path)
    df_ready = transform_data(df)
    X_train, X_test, y_train, y_test = train_test_data(df_ready)

    # model = LogisticRegression()
    # model.fit(X_train, y_train, iterations=100)
    # print(model.score)
    # # print(model.score())

    y_train = y_train["gesture_id"].values.reshape(-1, 1)
    y_test = y_test["gesture_id"].values.reshape(-1, 1)

    model = MultinomialLogisticRegression()

    model.fit(X_train, y_train, iterations=1000, save_loss=True)
    accuracy = model.predict_score(X_test, y_test)
    print(accuracy * 100)
