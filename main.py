import torch
from torch.utils.data import sampler, DataLoader
import matplotlib.pyplot as plt
import pickle
import psutil

import settings
from text_denoiser import denoise_text, replace_contractions
from TextDataset import TextDataset
from data_generators.CbowDataGenerator import CbowDataGenerator
from data_generators.NGramDataGenerator import NGramDataGenerator
from data_generators.GlobalNGramDataGenerator import GlobalNGramDataGenerator
from Vocabulary import Vocabulary
from CorpusStorage import CorpusStorage
from Models.NGramLanguageModeler import NGramLanguageModeler
from QuestionsParser import QuestionsParser
from Questionnaire import Questionnaire
from trainer_helpers import find_hyperparams, train_global_n_gram, train_n_gram
import utils


def write_generator(data_generator, path):
    with open(path, 'wb') as f:
        pickle.dump(data_generator, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('writing finished')
   
def main():
    # if settings.USE_GPU and torch.cuda.is_available():
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')

    # print('using device:', device)

    # with open('dataset/test.txt', 'r') as file:
    #     sample = file.read()
    # corpus_storage = CorpusStorage(sample)
    # with open(settings.test_corpus_storage_path, 'wb') as f:
    #     pickle.dump(corpus_storage, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(settings.test_corpus_storage_path, 'rb') as f:
        corpus_storage = pickle.load(f)

    vocabulary = Vocabulary(corpus_storage.corpus)

    # with open(settings.test_vocabulary_path, 'wb') as f:
    #     pickle.dump(vocabulary, f, protocol=pickle.HIGHEST_PROTOCOL)

    questions_parser = QuestionsParser(vocabulary, settings.test_questions_filename, [])
    # with open(settings.test_questions_parser_path, 'wb') as f:
    #     pickle.dump(questions_parser, f, protocol=pickle.HIGHEST_PROTOCOL)      

    cbow_data_generator = CbowDataGenerator(vocabulary, corpus_storage.corpus, window_size=5)
    pass
    # write_generator(cbow_data_generator, settings.test_cbow_data_generator_path)
    # n_gram_data_generator = NGramDataGenerator(vocabulary, corpus_storage.corpus, window_size=5)
    # write_generator(n_gram_data_generator, settings.test_n_gram_data_generator_path)
    # print('n-gram done')
    # global_n_gram_data_generator = GlobalNGramDataGenerator(vocabulary, corpus_storage.corpus, window_size=5)
    # write_generator(global_n_gram_data_generator, settings.test_global_n_gram_data_generator_path)
    # print('global-n-gram done')
    # # print('train done')
    # step_size = 2
    # best_val, best_model, best_stats, results = find_hyperparams(5, train_n_gram, vocabulary, 
    # settings.embedding_dim, device, step_size)
    # if settings.USE_GPU and torch.cuda.is_available():
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')

    # d = dict(psutil.virtual_memory()._asdict())
    # print(d)

    # with open(settings.test_n_gram_data_generator_path, 'rb') as f:
    #     data_generator = pickle.load(f)
    # a = data_generator.vocabulary.vocab_size
    # d = dict(psutil.virtual_memory()._asdict())
    # print(d)

    # with open(settings.test_questions_parser_path, 'rb') as f:
    #     questions_parser = pickle.load(f)
    # b = questions_parser.vocabulary.vocab_size
    # lr = 10**(-1.5)
    # gamma = 10**(-0.05)
    # step_size = 15
    # model_path = 'models_weights/n_gram'
    # train_losses, val_accuracies, model = train_n_gram(data_generator, questions_parser, settings.embedding_dim, device, lr,
    # gamma, step_size, model_path)



if __name__ == '__main__':
    main()