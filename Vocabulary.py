import nltk
from words_normalizer import normalize


class Vocabulary():
    def __init__(self, corpus):
        self.index_word = {}
        self.word_index = {}
        self.vocab_size = 0
        self.__process_corpus_and_fill_in_maps(corpus)
        self.all_words_indices = list(self.index_word.keys())

    def get_idx(self, word):
        if word not in self.word_index:
            self.__add_word(word)
        idx = self.word_index[word]
        return idx

    def get_word(self, idx):
        word = self.index_word[idx]
        return word

    def __process_corpus_and_fill_in_maps(self, corpus):
        counter = 0
        for normalized_words in corpus:
            for normalized_word in normalized_words:
                if normalized_word not in self.word_index:
                    self.index_word[counter] = normalized_word
                    self.word_index[normalized_word] = counter
                    counter += 1
            self.vocab_size = counter
    
    def __add_word(self, word):
        print(word)
        self.word_index[word] = self.vocab_size
        self.index_word[self.vocab_size] = word
        self.vocab_size += 1
