import numpy as np
import numpy.random as random
import torch
from torch.nn import CosineSimilarity


class Questionnaire():
    def __init__(self, vocabulary, model, questions_filename, meta_indices, device):
        self.vocabulary = vocabulary
        self.model = model
        self.questions_filename = questions_filename
        self.meta_indices = meta_indices
        self.device = device
        self.words_count_in_line = 4
        self.cosine_similarity = CosineSimilarity(dim=1)

        self.lines = self.__get_lines_from_file()
        self.non_meta_indices = self.__get_non_meta_indices()
        self.count_questions = len(self.lines) - len(self.meta_indices)
        indices = random.choice(self.non_meta_indices, 
        self.count_questions, replace=False)
        first_half = self.count_questions // 2
        val_indices = indices[:first_half]
        test_indices = indices[first_half:]
        self.val_questionnaire = self.__get_questionnaire(val_indices)
        self.test_questionnaire = self.__get_questionnaire(test_indices)
        self.__vocabulary_embeddings = self.__get_vocabulary_embeddings()

    def __get_non_meta_indices(self):
        # add fake meta_idx
        exclude_ids = self.meta_indices + [len(self.lines)]
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

    def __get_lines_by_indices(self, indices):
        lines = [self.lines[idx] for idx in indices]
        return lines

    def __get_questionnaire(self, indices):
        lines = self.__get_lines_by_indices(indices)
        questionnaire = torch.zeros((len(lines), self.words_count_in_line, 
        self.model.embedding_dim), device=self.device, dtype=torch.float32)
        for i, line in enumerate(lines):
            words = self.__get_words_from_line(line)
            words_indices = [self.vocabulary.get_idx(word) for word in words]
            words_indices = torch.LongTensor(words_indices, device=self.device)
            with torch.no_grad():
                embeddings = self.model.embeddings(words_indices)
                questionnaire[i] = embeddings

        return questionnaire

    def __get_vocabulary_embeddings(self):
        vocabulary_embeddings = None
        all_words_indices = torch.LongTensor(self.vocabulary.all_words_indices, device=self.device)
        with torch.no_grad():
            vocabulary_embeddings = self.model.embeddings(all_words_indices)
            
        return vocabulary_embeddings

    def __check_accuracy(self, questionnaire):
        expected_answers = questionnaire[:, 3]
        real_answers = torch.zeros_like(expected_answers)
        computed_embeds = questionnaire[:, 1] - questionnaire[:, 0] + questionnaire[:, 2]
        for i, embedding in enumerate(computed_embeds):
            embedding = embedding.view((1, -1))
            distances = self.cosine_similarity(embedding, self.__vocabulary_embeddings)
            # you have to check it again
            argmax = distances.argmax()
            top_idx = argmax.item()
            real_answers[i] = self.__vocabulary_embeddings[top_idx]
        
        correct_answers_count = 0
        for i, expected_answer in enumerate(expected_answers):
            if expected_answer.equal(real_answers[i]):
                correct_answers_count += 1

        accuracy = correct_answers_count / len(real_answers)
        return accuracy
            
    def check_val_accuracy(self):
        val_accuracy = self.__check_accuracy(self.val_questionnaire)
        return val_accuracy

    def check_test_accuracy(self):
        test_accuracy = self.__check_accuracy(self.test_questionnaire)
        return test_accuracy

    def get_analogy(self, target_word, positive_word, negative_word, top=5):
        target_idx = self.vocabulary.get_idx(target_word)
        positive_idx = self.vocabulary.get_idx(positive_word)
        negative_idx = self.vocabulary.get_idx(negative_word)
        target_embedding = self.__vocabulary_embeddings[target_idx]
        positive_embedding = self.__vocabulary_embeddings[positive_idx]
        negative_embedding = self.__vocabulary_embeddings[negative_idx]
        perfect_embedding = target_embedding - negative_embedding + positive_embedding
        perfect_embedding = perfect_embedding.view((1, -1))
        distances = self.cosine_similarity(perfect_embedding, 
        self.__vocabulary_embeddings)
        argmax_indices = distances.sort(descending=True).indices
        top_argmax_indices = argmax_indices[:top]
        analogies = [self.vocabulary.get_word(idx.item()) for idx in top_argmax_indices]
        return analogies, distances