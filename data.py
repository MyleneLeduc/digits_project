from torch.utils.data import Dataset
import pandas as pd
import os
import torch

class DigitsDataset(Dataset):
    def __init__(self, file_name):
        file_path = os.path.join('data', file_name)
        columns = [i for i in range(65)]
        df = pd.read_csv(file_path, names = columns, header=0)
        self.X = df.iloc[:,0:64]
        self.y = df.iloc[:,-1]
        self.X = torch.FloatTensor(self.X.to_numpy())
        self.y = torch.FloatTensor(self.y.to_numpy())
    def __len__(self):
        return len(self.X)
    def __getitem__(self):
        return self.X[i], self.y[i]
