import os
import shutil
import tempfile
import zipfile

import patoolib

from base_extractor import BaseDataExtractor
from utils.logger import logger


class FileSystemExtractor(BaseDataExtractor):
    """
    File System Data Pipeline extractor class. Extracts data from zip file and copies it
    to target folder.
    """

    def __init__(self, source_path, target_path):
        self._source_path = source_path
        self._target_path = target_path
        self.validate()
        super().__init__()

    def extract(self):
        logger.debug("Starting Data Extraction Process")
        logger.info(f"Extracting data from {self.source} to {self.target}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            logger.debug(f"Extracting data to tmp dir {tmp_dir}")
            with zipfile.ZipFile(self.source, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
                small_zips = os.listdir(tmp_dir)
                for small_zip in small_zips:
                    if small_zip.endswith(".rar"):
                        folder_name = small_zip.rstrip(".rar")
                        small_zip_path = os.path.join(tmp_dir, small_zip)
                        small_zip_out_path = os.path.join(self.target, folder_name)
                        if os.path.exists(small_zip_out_path):
                            shutil.rmtree(small_zip_out_path)
                        os.mkdir(small_zip_out_path)
                        patoolib.extract_archive(small_zip_path, outdir=small_zip_out_path)

    def validate(self):
        if not self.source.endswith(".zip"):
            logger.error(f"Source is not a zip file. {self.source}")
            raise Exception(f"Source is not a zip file. {self.source}")

        for path in [self.source, self.target]:
            if not os.path.exists(path):
                logger.error(f"Path {path} not found")
                raise Exception(f"Path {path} not found")


if __name__ == '__main__':
    extractor = FileSystemExtractor("/home/j3/Desktop/gesture-recognition/data/raw/uWaveGestureLibrary.zip",
                                    "/home/j3/Desktop/gesture-recognition/data/extracted")

    extractor.extract()
