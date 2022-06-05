from abc import ABCMeta, abstractmethod


class BaseDataExtractor:
    """
    Class for extracting data from raw format/input and add to the data lake
    """

    def __int__(self, source_path: str, target_path: str):
        self._source_path = source_path
        self._target_path = target_path

    @property
    def source(self):
        return self._source_path

    @property
    def target(self):
        return self._target_path
    
    @property
    def type(self):
        return self.__class__.__name__

    @abstractmethod
    def extract(self):
        pass

    @abstractmethod
    def validate(self):
        pass
