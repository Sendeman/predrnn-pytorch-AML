import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path 
import cv2
from torch import Tensor, save
import re

class KTHDatasetRNN(Dataset):
    def __init__(self, root_dir: Path = Path("dataset")/"KTH_data_latent", transform=None, max_frame_size: int = 10):
        """_summary_

        Args:
            root_dir (Path, optional): directory of latent space data. Defaults to Path("dataset")/"KTH_data_latent".
            transform (_type_, optional): _description_. Defaults to None.
        """
        self.root_dir = root_dir 
        self.transform = transform
        self.frames = []
        self.video_files = set()
        self.data = {} #dictionary of video file names and their corresponding frames
        self.max_frame_size = max_frame_size

        #list of sorted frame paths
        self.videos = []
        self.load_data()

    def load_data(self):
        # Load the data from the directory and populate self.data and self.labels
        for action in self.root_dir.iterdir():
            if action.is_dir():
                for frame in action.iterdir():
                    if frame.is_file() and frame.suffix == '.pt':
                        self.frames.append(frame)
        
        #get all video file names from frame files
        for frame in self.frames:
            index = re.sub(r'_frame_\d+', '', frame.stem)
            self.video_files.add(index)

        get_framenum = lambda pth: int(re.search(r'_(\d+)', pth.stem).group(1))
        #get all frames for each video file
        for video_file in self.video_files:
            frames = []
            for frame in self.frames:
                if video_file in frame.stem:
                    frames.append(frame)

            #sort frames by frame number
            frames.sort(key=get_framenum)
            self.data[video_file] = frames

    # def __len__(self):
    #     return len(self.frames)

    def __getitem__(self, idx, sequence_length: int = 10):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label
    
if __name__ == "__main__":
    dataset = KTHDatasetRNN()
    print(dataset.data)
