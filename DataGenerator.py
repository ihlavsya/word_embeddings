import numpy as np
import sys
from itertools import chain


class DataGenerator():
    def __init__(self, vocabulary, window_size, is_connection_map=False):
        self.window = window_size
        self.vocabulary = vocabulary
        self.half_window = self.window // 2
        if is_connection_map:
            self.connection_map = self.__get_connection_map()

    def generate_n_gram_training_data(self):
        training_data = []
        for i, sentence in enumerate(self.vocabulary.corpus):
            sentence_len = len(sentence)
            if sentence_len == 1:
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
    
    def generate_cbow_training_data(self):
        training_data = []
        for i, sentence in enumerate(self.vocabulary.corpus):
            sentence_len = len(sentence)
            if sentence_len < self.window:
                continue
            for start in range(sentence_len - self.window + 1):
                end = self.window + start
                middle_idx = start + self.half_window
                target = self.vocabulary.get_idx(sentence[middle_idx])
                context_range = concatenated = chain(range(start, middle_idx), range(middle_idx + 1, end))
                context = [self.vocabulary.get_idx(sentence[k]) for k in context_range]
                sample = (context, target)
                training_data.append(sample)
        
        return training_data

    def generate_global_n_gram_training_data(self):
        half_windows_count = self.__fill_in_connection_map_and_get_half_windows_count()  
        tf_idfs_map = {}
        min_tf_idf = sys.maxsize
        max_tf_idf = - sys.maxsize - 1
        for key, value in self.connection_map.items():
            context, target = key
            context = self.vocabulary.get_idx(context)
            target = self.vocabulary.get_idx(target)
            tf = value[1]/self.half_window
            eps = 1e-6
            idf = np.log(half_windows_count / (value[0] + eps))
            tf_idf = tf * idf
            if tf_idf < min_tf_idf:
                min_tf_idf = tf_idf
            if tf_idf > max_tf_idf:
                max_tf_idf = tf_idf
            key = (context, target)
            if key not in tf_idfs_map:
                tf_idfs_map[key] = tf_idf       

        tf_idfs = np.array(list(tf_idfs_map.values()), dtype=np.float32)
        normalized_tf_idfs = (tf_idfs - min_tf_idf) / (max_tf_idf - min_tf_idf)
        training_data = []
        for i, key in enumerate(tf_idfs_map):
            context, target = key
            tf_idf = normalized_tf_idfs[i]
            sample = ([context, target], tf_idf)
            training_data.append(sample)
        return training_data

    def __fill_in_connection_map_and_get_half_windows_count(self):
        half_windows_count = 0
        for i, sentence in enumerate(self.vocabulary.corpus):
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
                        number_of_windows_containing_pair = self.connection_map[key][0] + 1
                        freq_pair_in_window = self.connection_map[key][1] + 1
                        self.connection_map[key] = (number_of_windows_containing_pair, freq_pair_in_window)

                for start in range(1, self.half_window + 1):
                    idx = j + start
                    if idx < sentence_len:
                        target = sentence[idx]
                        key = (context, target)
                        number_of_windows_containing_pair = self.connection_map[key][0] + 1
                        freq_pair_in_window = self.connection_map[key][1] + 1
                        self.connection_map[key] = (number_of_windows_containing_pair, freq_pair_in_window)

        return half_windows_count

    def __get_connection_map(self):
        connection_map = {}
        normalized_words = [item for sublist in self.vocabulary.corpus for item in sublist]
        normalized_words_len = len(normalized_words)
        for i in range(normalized_words_len):
            for j in range(normalized_words_len):
                key = (normalized_words[i], normalized_words[j])
                if key not in connection_map:
                    connection_map[key] = (0, 0)

        return connection_map
