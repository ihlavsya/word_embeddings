import os
import torch
import numpy as np


class ModelTrainer():
    def __init__(self, model, optimizer, scheduler, loss_function, 
    data_loader, questionnaire, path=None):
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model = model
        self.scheduler = scheduler
        self.data_loader = data_loader
        self.questionnaire = questionnaire
        self.path = path
        
    def save_checkpoint(self, epoch, loss):
        checkpoint_path = os.path.join(self.path, '%d.pt'%epoch)
        checkpoint_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint_dict, checkpoint_path)

    def train(self, epochs=10, verbose=False, batch_size=32): 
        train_losses = []
        val_accuracies = []
        # return train_losses and val_accuracies
        # i don`t see better solution
        for t in range(epochs):
            total_loss = 0
            for X, y in self.data_loader:
                self.model.zero_grad()
                log_prob = self.model(X)
                loss = self.loss_function(log_prob, y)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += (loss.item())/batch_size
            if t % 5 == 0:
                self.questionnaire.update_model_and_embeddings(self.model)
                val_accuracy = self.questionnaire.check_val_accuracy()
                val_accuracies.append(val_accuracy)
                print(val_accuracy)
            train_losses.append(total_loss)
            if verbose:
                print('epoch:', t)
                print('loss:', total_loss)
            if self.path is not None:
                self.save_checkpoint(t, total_loss)
                
        return train_losses, val_accuracies