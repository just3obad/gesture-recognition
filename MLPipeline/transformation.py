from sklearn.pipeline import Pipeline
from utils.logger import logger
import pandas as pd
import itertools


def transform_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    # prepare feature vectores
    # sample the data

    gestures_df = data_frame
    # Add feature vector size
    gestures_df["feature_vector_size"] = gestures_df.X.apply(lambda x: len(x))
    # Drop outliers. data points with feature size < 25
    gestures_df = gestures_df[gestures_df["feature_vector_size"] > 25]
    # Select needed features for model
    gestures_df = gestures_df[["gesture_id", "X", "Y", "Z"]]
    # zip x y z data points and flatten the list
    # TODO: sample the data into a fixed feature vector size
    gestures_df["features"] = gestures_df.apply(lambda row: list(itertools.chain(*zip(row.X, row.Y, row.Z))), axis=1)


    # gestures_df_train_test

    print(gestures_df.shape)
    print(gestures_df.head())
    print(gestures_df.features)

    return gestures_df


def train_test_data(data_frame: pd.DataFrame):
    pass


if __name__ == '__main__':
    data_path = "/home/j3/Desktop/gesture-recognition/data/clean/gestures_clean.pkl"
    df = pd.read_pickle(data_path)

    transform_data(df)
