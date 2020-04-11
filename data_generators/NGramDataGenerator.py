import numpy as np
import sys
from itertools import chain, product

from DataGenerator import DataGenerator


class NGramDataGenerator(DataGenerator):
    def __init__(self, vocabulary, corpus, window_size):
        super().__init__(vocabulary, corpus, window_size)
        self._training_data = self._generate_training_data(corpus)

    def _generate_training_data(self, corpus):
        training_data = []
        for i, sentence in enumerate(corpus):
            sentence_len = len(sentence)
            if i % 100 == 0:
                print('iteration: %d' %i)
            if (sentence_len == 1):
                continue
            for j, word in enumerate(sentence):
                context = self.vocabulary.get_idx(word)
                for start in range(1, self.half_window + 1):
                    idx = j - start
                    if idx >= 0:
                        target = self.vocabulary.get_idx(sentence[idx])
                        sample = (context, target)
                        training_data.append(sample)

                for start in range(1, self.half_window + 1):
                    idx = j + start
                    if idx < sentence_len:
                        target = self.vocabulary.get_idx(sentence[idx])
                        sample = (context, target)
                        training_data.append(sample)      
        return training_data
    
    def get_training_data(self):
        return self._training_data