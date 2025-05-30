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
        """
        Args:
            file_paths (np.array): Array of file paths to the data files.
            action_labels (np.array): Array of action labels corresponding to the data files.
            transform (callable, optional): Optional transform to be applied on a sample. We use this for image augmentation (flipping).
        """
        self.transform = transform
        self.paths = file_paths
        self.labels = action_labels


    def load_data(self, file_loc):
        """
        Load the data from the given file location. We can't fully load everything to memory, so we load it on the fly. It's a bit slower, but at least it runs :-).
        Args:
            file_loc (str): Path to the data file.
        Returns:
            torch.Tensor: Loaded data as a PyTorch tensor.
        """
        return torch.load(file_loc)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to fetch.
        Returns:
            tuple: (sample, label) where sample is the loaded data and label is the corresponding action label.
        """
        
        sample = torch.load(self.paths[idx])
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