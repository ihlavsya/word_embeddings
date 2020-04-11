import numpy as np
import numpy.random as random
import torch
from torch.nn import CosineSimilarity


class Questionnaire():
    def __init__(self, questions_parser, model):
        self.questions_parser = questions_parser
        self.model = model
        self.vocabulary = self.questions_parser.vocabulary
        self.cosine_similarity = CosineSimilarity(dim=1)
        a = self.questions_parser.meta_indices
        indices = random.choice(self.questions_parser.non_meta_indices, 
        self.questions_parser.questions_count, replace=False)
        first_half = self.questions_parser.questions_count // 2
        self.val_indices = indices[:first_half]
        self.test_indices = indices[first_half:]
        self.val_questionnaire = self.__get_questionnaire(self.val_indices)
        self.test_questionnaire = self.__get_questionnaire(self.test_indices)
        self.__vocabulary_embeddings = self.__get_vocabulary_embeddings()

    def update_model_and_embeddings(self, model):
        self.model = model
        self.val_questionnaire = self.__get_questionnaire(self.val_indices)
        self.test_questionnaire = self.__get_questionnaire(self.test_indices)
        self.__vocabulary_embeddings = self.__get_vocabulary_embeddings()

    def __get_questionnaire(self, indices):
        # maybe you can simply vectorize it?
        word_to_idx_table = self.questions_parser.get_line_words_indices(indices)
        word_to_idx_table = torch.tensor(word_to_idx_table, dtype=torch.long)
        with torch.no_grad():
            questionnaire = self.model.embeddings(word_to_idx_table)

        return questionnaire

    def __get_vocabulary_embeddings(self):
        vocabulary_embeddings = None
        all_words_indices = torch.tensor(self.vocabulary.all_words_indices, device=self.model.device)
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