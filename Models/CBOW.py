import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, half_window, batch_size):
        super(CBOW, self).__init__()
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(half_window * 2 * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((self.batch_size, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs