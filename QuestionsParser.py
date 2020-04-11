import numpy as np


class QuestionsParser():
    def __init__(self, vocabulary, questions_filename, meta_indices):
        self.vocabulary = vocabulary
        self.questions_filename = questions_filename
        self.meta_indices = meta_indices
        self.words_count_in_line = 4

        self.__lines = self.__get_lines_from_file()
        self.non_meta_indices = self.__get_non_meta_indices()
        self.questions_count = len(self.__lines) - len(self.meta_indices)
        self.__words_indices = self.__get_all_words_indices()

    def __get_non_meta_indices(self):
        # add fake meta_idx
        exclude_ids = self.meta_indices + [len(self.__lines)]
        indices = []
        start = 0
        for exclude_ind in exclude_ids:
            partial_range = range(start, exclude_ind)
            indices.extend(partial_range)
            start = exclude_ind + 1

        return indices

    def __get_lines_from_file(self):
        lines = None
        with open(self.questions_filename, 'r') as file:
            lines = file.readlines()

        return lines

    def __get_words_from_line(self, line):
        words = line.strip('\n').split(' ')
        return words

    def __get_all_words_indices(self):
        all_words_indices = np.zeros((len(self.__lines), self.words_count_in_line), dtype=np.int)
        for i in self.non_meta_indices:
            line = self.__lines[i]
            words = self.__get_words_from_line(line)
            words_indices = [self.vocabulary.get_idx(word) for word in words]
            all_words_indices[i] = words_indices

        return all_words_indices

    def get_line_words_indices(self, indices):
        words_indices = self.__words_indices[indices]
        return words_indices
