import torch.nn as nn
import torch.nn.functional as F

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(NGramLanguageModeler, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        self.linear1 = nn.Linear(self.embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, input):
        embed = self.embeddings(input)
        out = F.relu(self.linear1(embed))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs  