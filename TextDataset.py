import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data
        self.data_length = len(self.data)

  def __len__(self):
        'Denotes the total number of samples'
        return self.data_length

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        sample = self.data[index]

        # Load data and get label
        X = torch.tensor(sample[0])
        y = sample[1]

        return X, y