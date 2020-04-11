import psutil
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import sampler, DataLoader

import settings
from TextDataset import TextDataset
from IterableGlobalNGramDataset import IterableGlobalNGramDataset
from ModelTrainer import ModelTrainer
from Models.GlobalNGramModel import GlobalNGramModel
from Models.NGramLanguageModeler import NGramLanguageModeler
from Models.CBOW import CBOW
from Questionnaire import Questionnaire
from DataGenerator import DataGenerator

def train_global_n_gram(data_generator, questions_parser, 
    embedding_dim, device, lr, gamma, step_size):
    tf_idfs_map, max_tf_idf = data_generator.get_training_data()
    vocabulary = data_generator.vocabulary
    global_ngram_dataset = IterableGlobalNGramDataset(vocabulary.all_words_indices, 
    tf_idfs_map, max_tf_idf, device)
    loader_train = DataLoader(global_ngram_dataset, batch_size=settings.batch_size, 
                          #sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)),
                          drop_last=True)

    loss_function = torch.nn.BCELoss()
    model = GlobalNGramModel(questions_parser.vocabulary.vocab_size, embedding_dim, 
    settings.half_window_size, settings.batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    questionnaire = Questionnaire(questions_parser, model)
    print('questionnaire has setup')
    trainer = ModelTrainer(model, optimizer, scheduler, loss_function, loader_train,
    questionnaire, model_path)
    print('setup completed')
    d = dict(psutil.virtual_memory()._asdict())
    print(d)
    train_losses, val_accuracies = trainer.train(epochs=settings.epochs, verbose=True, batch_size=settings.batch_size)
    return train_losses, val_accuracies, model


def train_cbow(training_data, questions_parser, embedding_dim, device, lr, 
gamma, step_size, model_path):
    NUM_TRAIN = len(training_data)
    text_dataset = TextDataset(training_data, device)
    loader_train = DataLoader(text_dataset, batch_size=settings.batch_size, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)),
                          drop_last=True)

    loss_function = torch.nn.NLLLoss()
    model = CBOW(questions_parser.vocabulary.vocab_size, settings.embedding_dim, 
    settings.half_window_size, settings.batch_size, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    questionnaire = Questionnaire(questions_parser, model)
    print('questionnaire has setup')
    trainer = ModelTrainer(model, optimizer, scheduler, loss_function, loader_train,
    questionnaire, model_path)
    print('setup completed')
    d = dict(psutil.virtual_memory()._asdict())
    print(d)
    train_losses, val_accuracies = trainer.train(epochs=settings.epochs, verbose=True, batch_size=settings.batch_size)
    return train_losses, val_accuracies, model


def train_n_gram(training_data, questions_parser,
    embedding_dim, device, lr, gamma, step_size, model_path):
    NUM_TRAIN = len(training_data)
    text_dataset = TextDataset(training_data, device)
    loader_train = DataLoader(text_dataset, batch_size=settings.batch_size, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)),
                          drop_last=True)
    loss_function = torch.nn.NLLLoss()
    model = NGramLanguageModeler(questions_parser.vocabulary.vocab_size, settings.embedding_dim, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    questionnaire = Questionnaire(questions_parser, model)
    trainer = ModelTrainer(model, optimizer, scheduler, loss_function, loader_train,
    questionnaire, model_path)
    train_losses, val_accuracies = trainer.train(epochs=settings.epochs, verbose=True, batch_size=settings.batch_size)
    return train_losses, val_accuracies, model


def find_hyperparams(max_count, rate_r_min, rate_r_max, gamma_r_min, gamma_r_max, train_function, data_generator,
 questions_parser, embedding_dim, device, step_size):
    best_loss = 2000
    best_model = None
    best_stats = None
    results = {}
    for i in range(max_count):
        print('iteration %d' % i)
        rate_r = np.random.uniform(rate_r_min, rate_r_max)
        gamma_r = np.random.uniform(gamma_r_min, gamma_r_max)
        lr = 10**rate_r
        gamma = 10**gamma_r 
        # Train the network
        print('rate_r: %f, gamma_r: %f' % (rate_r, gamma_r))
        try:
            train_losses, val_accuracies, model = train_function(data_generator, questions_parser, 
            embedding_dim, device,
            lr, gamma, step_size, None)
        except RuntimeError as e:
            print(e)
            continue
        # we take here last accuracy for simplicity
        val_accuracy = val_accuracies[-1]
        train_loss = train_losses[-1]
        print('validation accuracy: %f,  train_loss: %f, rate_r: %f, gamma_r: %f' % (val_accuracy, train_loss, rate_r, gamma_r))
        results[(rate_r, gamma_r)] = (val_accuracy, lr, gamma)
        if best_loss > train_loss:
            best_loss = train_loss
            best_val = val_accuracy
            best_model = model
            best_stats = (train_losses, val_accuracies)

    return best_val, best_model, best_stats, results
    
