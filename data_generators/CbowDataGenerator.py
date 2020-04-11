import numpy as np
import sys
from itertools import chain, product

from DataGenerator import DataGenerator


class CbowDataGenerator(DataGenerator):
    def __init__(self, vocabulary, corpus, window_size):
        super().__init__(vocabulary, corpus, window_size)
        self._training_data = self._generate_training_data(corpus)

    def _generate_training_data(self, corpus):
        training_data = []
        for sentence in corpus:
            sentence_len = len(sentence)
            if sentence_len < self.window:
                continue
            for start in range(sentence_len - self.window + 1):
                end = self.window + start
                middle_idx = start + self.half_window
                target = self.vocabulary.get_idx(sentence[middle_idx])
                context_range = chain(range(start, middle_idx), range(middle_idx + 1, end))
                context = [self.vocabulary.get_idx(sentence[k]) for k in context_range]
                sample = (context, target)
                training_data.append(sample)
        
        return training_data

    def get_training_data(self):
        return self._training_data
