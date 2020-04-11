import numpy as np
import sys
from itertools import chain, product

from DataGenerator import DataGenerator


class GlobalNGramDataGenerator(DataGenerator):
    def __init__(self, vocabulary, corpus, window_size):
        super().__init__(vocabulary, corpus, window_size)
        self.__connection_map, self.__half_windows_count = self.__get_connection_map_and_get_half_windows_count(corpus)
        self._training_data = self._generate_training_data()

    def _generate_training_data(self):
        tf_idfs_map = {}

        max_tf_idf = - sys.maxsize - 1
        for key, value in self.__connection_map.items():
            context, target = key
            context = self.vocabulary.get_idx(context)
            target = self.vocabulary.get_idx(target)
            tf = value[1]/self.half_window
            eps = 1e-6
            idf = np.log(self.__half_windows_count / (value[0] + eps))
            tf_idf = tf * idf
            if tf_idf > max_tf_idf:
                max_tf_idf = tf_idf
            key = (context, target)
            if key not in tf_idfs_map:
                tf_idfs_map[key] = tf_idf       

        # tf_idfs = np.array(list(tf_idfs_map.values()), dtype=np.float32)
        # considering the fact that min_tf_idf will always be 0,
        # you can rewrite this as (tf_idfs - min_tf_idf) / (max_tf_idf -min_tf_idf)
        # normalized_tf_idfs = tf_idfs / max_tf_idf
        # you have also rewrite this part
        # training_data = []
        # for i, key in enumerate(tf_idfs_map):
        #     context, target = key
        #     tf_idf = normalized_tf_idfs[i]
        #     sample = ([context, target], tf_idf)
        #     training_data.append(sample)
        return tf_idfs_map, max_tf_idf

    def __get_connection_map_and_get_half_windows_count(self, corpus):
        connection_map = {}
        half_windows_count = 0
        for sentence in corpus:
            sentence_len = len(sentence)
            if sentence_len == 1:
                continue
            half_windows_count += 2 * (sentence_len - 1)
            for j, word in enumerate(sentence):
                context = word
                for start in range(1, self.half_window + 1):
                    idx = j - start
                    if idx >= 0:
                        target = sentence[idx]
                        key = (context, target)
                        if key not in connection_map:
                            connection_map[key] = (0., 0.)
                        number_of_windows_containing_pair = connection_map[key][0] + 1
                        freq_pair_in_window = connection_map[key][1] + 1
                        connection_map[key] = (number_of_windows_containing_pair, freq_pair_in_window)

                for start in range(1, self.half_window + 1):
                    idx = j + start
                    if idx < sentence_len:
                        target = sentence[idx]
                        key = (context, target)
                        if key not in connection_map:
                            connection_map[key] = (0., 0.)
                        number_of_windows_containing_pair = connection_map[key][0] + 1
                        freq_pair_in_window = connection_map[key][1] + 1
                        connection_map[key] = (number_of_windows_containing_pair, freq_pair_in_window)

        return connection_map, half_windows_count

    def get_training_data(self):
        return self._training_data
