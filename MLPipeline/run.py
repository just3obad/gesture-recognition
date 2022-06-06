from preprocessing import FileSystemDataPreprocessor
from training import train_logistic_regression, train_svm, train_random_forrest, train_custom_logistic_regression
from transformation import transform_data, train_test_data


def main():
    source = "/home/j3/Desktop/gesture-recognition/data/extracted"

    processor = FileSystemDataPreprocessor(source)
    df = processor.start()
    df_ready = transform_data(df)
    X_train, X_test, y_train, y_test = train_test_data(df_ready)

    train_logistic_regression(X_train, X_test, y_train, y_test)
    train_svm(X_train, X_test, y_train, y_test)
    train_random_forrest(X_train, X_test, y_train, y_test)
    train_custom_logistic_regression(X_train, X_test, y_train, y_test)

    # After training, models can be validated on unseen data sets


if __name__ == '__main__':
    main()
