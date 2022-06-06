import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from models.logistic_regression import MultinomialLogisticRegression
from utils.logger import logger

model_registry = "/home/j3/Desktop/gesture-recognition/local_models_registry"


def train_logistic_regression(X_train, X_test, y_train, y_test):
    logger.info(f"Training LogisticRegression")
    model = LogisticRegression(multi_class='multinomial')
    model.fit(X_train, y_train)
    logger.info(f"Accuracy%: {model.score(X_test, y_test) * 100}")
    joblib.dump(model, os.path.join(model_registry, "logistic_regression_model.pkl"))
    return model


def train_svm(X_train, X_test, y_train, y_test):
    model = SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo')
    model.fit(X_train, y_train)
    logger.info(f"Accuracy%: {model.score(X_test, y_test) * 100}")
    joblib.dump(model, os.path.join(model_registry, "svm_model.pkl"))
    return model


def train_random_forrest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)
    logger.info(f"Accuracy%: {model.score(X_test, y_test) * 100}")
    joblib.dump(model, os.path.join(model_registry, "random_forrest_model.pkl"))
    return model


def train_custom_logistic_regression(X_train, X_test, y_train, y_test):
    model = MultinomialLogisticRegression()
    model.fit(X_train, y_train)
    logger.info(f"Accuracy%: {model.predict_score(X_test, y_test) * 100}")
    joblib.dump(model, os.path.join(model_registry, "custom_logistic_regression_model.pkl"))
    return model

# if __name__ == '__main__':
#     data_path = "/home/j3/Desktop/gesture-recognition/data/clean/gestures.pkl"
#     df = pd.read_pickle(data_path)
#     df_ready = transform_data(df)
#     X_train, X_test, y_train, y_test = train_test_data(df_ready)
#
#     train_logistic_regression(X_train, X_test, y_train, y_test)
#     train_svm(X_train, X_test, y_train, y_test)
#     train_random_forrest(X_train, X_test, y_train, y_test)
