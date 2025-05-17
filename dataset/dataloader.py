"""
Classes for the dataset (later used for the dataloaders) of the KTH dataset.
Author: Paul Verhoeven & Sander Geurts
Date: 16-05-2025 (dd-mm-yyyy)
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path


class KTHDataset(Dataset):
    def __init__(self, file_paths:np.array, action_labels:np.array, transform=None):
        self.transform = transform
        self.paths = file_paths
        self.labels = action_labels


    def load_data(self, file_loc):
        return torch.load(file_loc)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        sample = self.load_data(self.paths[idx])
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label
   
    
class KTHDatasetRNN(Dataset):
    def __init__(self, root_dir: Path = Path("dataset\KTH_data"), transform=None):
        self.root_dir = root_dir 
        self.transform = transform
        self.videos = []
        self.load_data()

    def load_data(self):
        # Load the data from the directory and populate self.data and self.labels
        ...
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label