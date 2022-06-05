"""Various Data Preprocessors that transforms data from internal raw format and prepare it for the
modeling process. Optionally data processors can choose to store cleaned data in to data lakes
to enable further analysis and offline training"""

import pandas as pd
import re
from tqdm import tqdm
import os
from utils.logger import logger


class FileSystemDataPreprocessor:
    def __init__(self, source_path: str, target_path: str = "", save_data: bool = False):
        self._source_path = source_path
        self._target_path = target_path
        self._save_data = save_data
        self.validate()

    @property
    def source(self):
        return self._source_path

    @property
    def target(self):
        return self._target_path

    @property
    def save_data(self):
        return self._save_data

    def validate(self):
        """
        Validates preprocessor configurations
        """
        if not os.path.exists(self.source):
            logger.error(f"Invalid source path {self.source}")
            raise Exception(f"Invalid source path {self.source}")

        if self.save_data:
            if not os.path.exists(self.target):
                logger.error(f"Invalid target path {self.target} to save data")
                raise Exception(f"Invalid target path {self.target} to save data")

    def _process_folder(self, source_folder):
        columns = ["user_id", "day", "template", "file_name", "gesture_id", "repetition", "X", "Y", "Z"]
        df = pd.DataFrame(columns=columns)
        user_id = re.search(r"U\d", source_folder).group(0)
        day = re.search(r"\((\d)\)", source_folder).group(1)

        for txt_file in tqdm(os.listdir(source_folder)):
            if "Template_Acceleration" in txt_file:
                template = txt_file[0]
                gesture_id = re.search(r"Acceleration(\d)", txt_file).group(1)
                try:
                    repetition = re.search(r"Acceleration\d-(\d)", txt_file).group(1)
                except Exception as e:
                    repetition = -1

                txt_df = pd.read_csv(os.path.join(source_folder, txt_file), sep=" ", names=["X", "Y", "Z"])
                x = txt_df.X.to_numpy()
                y = txt_df.Y.to_numpy()
                z = txt_df.Z.to_numpy()

                tmp_df = pd.DataFrame([[user_id, day, template, txt_file, gesture_id, repetition, x, y, z]],
                                      columns=columns)
                df = pd.concat([df, tmp_df])

        return df

    def start(self):
        logger.info("Starting processing data")
        clean_df = pd.DataFrame()

        for folder_ in os.listdir(self.source):
            folder_path = os.path.join(self.source, folder_)
            logger.debug(f"Processing {folder_path}")
            df = self._process_folder(folder_path)
            clean_df = pd.concat([clean_df, df])

        if self.save_data:
            clean_df.to_pickle(os.path.join(self.target, "gestures.pkl"))
        return clean_df


if __name__ == '__main__':
    source = "/home/j3/Desktop/gesture-recognition/data/extracted"
    target = "/home/j3/Desktop/gesture-recognition/data/clean"

    processor = FileSystemDataPreprocessor(source, target, True)
    processor.start()
