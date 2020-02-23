import torch
import numpy as np


class ModelTrainer():
    def __init__(self, model, optimizer, loss_function, data_loader):
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model = model
        self.data_loader = data_loader

    def train(self, epochs=10, verbose=False, batch_size=32): 
        losses = []
        for t in range(epochs):
            total_loss = 0
            for X, y in self.data_loader:
                self.model.zero_grad()
                log_prob = self.model(X)
                loss = self.loss_function(log_prob, y)
                loss.backward()
                self.optimizer.step()
                total_loss += (loss.item())/batch_size
            losses.append(total_loss)
            if verbose:
                print('epoch:', t)
                print('loss:', total_loss)
        return losses