import numpy as np
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import sampler, DataLoader
import matplotlib.pyplot as plt
from itertools import chain

from TextDataset import TextDataset
from DataGenerator import DataGenerator
from text_denoiser import denoise_text, replace_contractions
from Vocabulary import Vocabulary
from Questionnaire import Questionnaire

from Models.NGramLanguageModeler import NGramLanguageModeler
from Models.CBOW import CBOW
from Models.GlobalNGramModel import GlobalNGramModel
from ModelTrainer import ModelTrainer


def train_n_gram(data_generator, embedding_dim):
    training_data = data_generator.generate_n_gram_training_data()
    NUM_TRAIN = len(training_data)
    text_dataset = TextDataset(training_data)
    loader_train = DataLoader(text_dataset, batch_size=32, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)),
                          drop_last=True)

    loss_function = torch.nn.NLLLoss()
    model = NGramLanguageModeler(data_generator.vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    trainer = ModelTrainer(model, optimizer, loss_function, loader_train)
    losses = trainer.train(epochs=40, verbose=True)
    return losses


def train_cbow(data_generator, embedding_dim):
    batch_size = 32
    training_data = data_generator.generate_cbow_training_data()
    NUM_TRAIN = len(training_data)
    text_dataset = TextDataset(training_data)
    loader_train = DataLoader(text_dataset, batch_size=batch_size, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)),
                          drop_last=True)

    loss_function = torch.nn.NLLLoss()
    model = CBOW(data_generator.vocabulary.vocab_size, embedding_dim, 2, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = ModelTrainer(model, optimizer, loss_function, loader_train)
    losses = trainer.train(epochs=10, verbose=True)
    return losses, model


def train_global_n_gram(data_generator, embedding_dim):
    batch_size=32
    training_data = data_generator.generate_global_n_gram_training_data()
    NUM_TRAIN = len(training_data)
    text_dataset = TextDataset(training_data)
    loader_train = DataLoader(text_dataset, batch_size=batch_size, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)),
                          drop_last=True)

    loss_function = torch.nn.BCELoss()
    model = GlobalNGramModel(data_generator.vocab_size, embedding_dim, 2, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = ModelTrainer(model, optimizer, loss_function, loader_train)
    losses = trainer.train(epochs=40, verbose=True)
    return losses


def get_filenames(directory_path):
    filenames = []
    for root, dirs, files in os.walk(directory_path):
        for name in files:
            if not name.startswith('.'):
                filenames.append(os.path.join(root, name))
    return filenames


def write_into_one_big_file(out_filename, in_filenames):
    prefix = ('From:', 'Subject:', 'Organization:', 'Lines:', 'In article')
    with open(out_filename, 'w') as outfile:
        for name in in_filenames:
            with open(name, 'r') as infile:
                text = []
                try:
                    for line in infile:
                        if not line.startswith(prefix):
                            text.append(line)
                except UnicodeDecodeError as e:
                    print(e)
                    print(name)
                    continue
                # remove author
                outfile.writelines(text[:-3])


def prepare_data():
    dir_path = 'dataset/20news-bydate'
    filenames = get_filenames(dir_path)
    write_into_one_big_file('dataset/train.txt', filenames)


def plot_losses(losses):
    plt.plot(losses)
    plt.show()

def train_everything_and_save(text):
    embedding_dim = 10
    data_generator = DataGenerator(text, window_size=5, is_connection_map=True)
    #losses1 = train_global_n_gram(data_generator, embedding_dim)
    print('1st loss done')
    #losses2 = train_n_gram(data_generator, embedding_dim)
    print('2nd loss done')
    losses3 = train_cbow(data_generator, embedding_dim)
    print('3rd loss done')

    #np.savetxt('losses1.txt', np.array(losses1), delimiter=',')
    plot_losses(losses3)
    #np.savetxt('losses2.txt', np.array(losses2), delimiter=',')
    #np.savetxt('losses3.txt', np.array(losses3), delimiter=',')


def main():
    # accuracy_filename = 'dataset/check_accuracy/questions-words.txt'
    # count_meta_lines = 15
    # meta_lines_indices = [0, 1, 508, 5034, 5900, 8368, 8875, 9868,
    # 10681, 12014, 13137, 14194, 15794, 17355, 18688, 19559]
    # count_question_lines = 19544
    # sample = None
    embedding_dim = 10
    with open('dataset/small_test.txt', 'r') as file:
        sample = file.read()

    denoised_sample = denoise_text(sample)
    text = replace_contractions(denoised_sample)

    vocab = Vocabulary(text)
    data_generator = DataGenerator(vocab, window_size=5, is_connection_map=True)
    losses, model = train_cbow(data_generator, embedding_dim)

    accuracy_filename = 'dataset/check_accuracy/questions.txt'
    meta_indices = []
    questionnaire = Questionnaire(vocab, model, accuracy_filename, meta_indices)
    accuracy = questionnaire.check_val_accuracy()
    print(accuracy)
    # result = questionnaire.get_analogy('atheism', 'christianity', 'atheist')


if __name__ == '__main__':
    main()