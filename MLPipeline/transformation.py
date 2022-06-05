import itertools

import pandas as pd

from sklearn.model_selection import train_test_split


def transform_data(data_frame: pd.DataFrame) -> pd.DataFrame:
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
    gestures_df = pd.concat(
        [gestures_df["gesture_id"].reset_index(drop=True), pd.DataFrame(gestures_df.features.tolist())], axis=1,
        ignore_index=True)
    gestures_df.rename(columns={0: "gesture_id"}, inplace=True)
    # Filling missing values
    gestures_df.fillna(0, inplace=True)
    return gestures_df


def train_test_data(data_frame: pd.DataFrame):
    gestures_X = data_frame.drop(["gesture_id"], axis=1)
    gestures_y = data_frame[["gesture_id"]]
    X_train, X_test, y_train, y_test = train_test_split(gestures_X, gestures_y, test_size=0.3, random_state=7)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    data_path = "/home/j3/Desktop/gesture-recognition/data/clean/gestures_clean.pkl"
    df = pd.read_pickle(data_path)
    df_ready = transform_data(df)
    print(df_ready.shape)
    print(df_ready.head())
