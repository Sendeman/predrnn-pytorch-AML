import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path


## TODO DATALOADER STUFFIES.

class KTHDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        # Load the data from the directory and populate self.data and self.labels
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
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