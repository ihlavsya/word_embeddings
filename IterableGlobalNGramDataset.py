import torch
from torch.utils.data import IterableDataset
from itertools import product


class IterableGlobalNGramDataset(IterableDataset):
    def __init__(self, vocabulary_ids, tf_idfs_map, max_tf_idf, device):
        super(IterableGlobalNGramDataset).__init__()
        self.vocabulary_ids = vocabulary_ids
        self.tf_idfs_map = tf_idfs_map
        self.max_tf_idf = max_tf_idf
        self.device = device
        self.groups = 100
        self.batch_size = len(self.vocabulary_ids) // self.groups

    def __iter__(self):
        # only single-process data loading, return the full iterator
        keys = product(self.vocabulary_ids, self.vocabulary_ids)
        for key in keys:
            value = 0.
            if key in self.tf_idfs_map:
                value = self.tf_idfs_map[key] / self.max_tf_idf
            X = torch.tensor(key, device=self.device)
            y = torch.tensor(value, device=self.device)
            yield X, y

