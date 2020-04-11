import nltk
from text_denoiser import denoise_text, replace_contractions
from words_normalizer import normalize


class CorpusStorage():
    def __init__(self, sample):
        text = denoise_text(sample)
        text = replace_contractions(text)
        self.corpus = self.__get_corpus(text)

    def __get_corpus(self, text):
        sentences = nltk.sent_tokenize(text)
        corpus = list()
        for i, sentence in enumerate(sentences):
            words = nltk.word_tokenize(sentence)
            normalized_words = normalize(words)
            corpus.append(normalized_words)
        return corpus
        