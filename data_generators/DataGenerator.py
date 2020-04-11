import numpy as np
import sys
from itertools import chain, product
from abc import ABC, abstractmethod


class DataGenerator(ABC):
    @abstractmethod
    def __init__(self, vocabulary, corpus, window_size):
        self.window = window_size
        self.vocabulary = vocabulary
        self.half_window = self.window // 2
        self._training_data = None

    @abstractmethod
    def get_training_data(self):
        raise NotImplementedError