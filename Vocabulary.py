import nltk
from words_normalizer import normalize


class Vocabulary():
    def __init__(self, text):
        self.text = text
        self.index_word = {}
        self.word_index = {}
        self.corpus = list()
        self.vocab_size = 0
        self.__process_text_and_fill_in_maps()
        self.all_words_indices = list(self.index_word.keys())

    def get_idx(self, word):
        idx = self.word_index[word]
        return idx

    def get_word(self, idx):
        word = self.index_word[idx]
        return word

    def __process_text_and_fill_in_maps(self):
        sentences = nltk.sent_tokenize(self.text)
        counter = 0
        for i, sentence in enumerate(sentences):
            words = nltk.word_tokenize(sentence)
            normalized_words = normalize(words)
            self.corpus.append(normalized_words)
            for normalized_word in normalized_words:
                if normalized_word not in self.word_index:
                    self.index_word[counter] = normalized_word
                    self.word_index[normalized_word] = counter
                    counter += 1
            self.vocab_size = counter
