import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GlobalNGramModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, half_window, batch_size):
        super(GlobalNGramModel, self).__init__()
        self.batch_size = batch_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(2 * embedding_dim, 128)
        self.linear2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((self.batch_size, -1))
        out = F.relu(self.linear1(embeds))
        out = self.sigmoid(self.linear2(out)).view((-1,))
        return out