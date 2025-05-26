import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class AutoencoderModel(nn.Module):
    def __init__(self, latent_dim:int=100, epochs:int = 150, batch_size:int = 64, learning_rate:float = 1e-3):
        """
        Autoencoder model for image reconstruction.
        Args:
            latent_dim (int): Dimension of the latent space.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
            learning_rate (float): Learning rate for the optimizer.
        """

        
        super(AutoencoderModel, self).__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.trained_epochs = 0

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1), # 1x120x160 -> 32 x 120 x 160
            nn.LeakyReLU(),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), # 32x120x160 -> 32x60x80
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 2x120x160 -> 4x60x80
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), # 2x120x160 -> 4x60x80
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU()
        )


        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),  # 32x15x20 -> 16x30x40
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),  # 16x30x40 -> 16x60x80
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),  # 16x60x80 -> 8x120x160
            nn.LeakyReLU(),

            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),  # 4x120x160 -> 1x120x160
            nn.Sigmoid()  # Pixels are within [0, 1]
        )


    def forward(self, x):
        """
        Forward pass through the autoencoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 120, 160).
        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, 1, 120, 160).
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

    def save(self, folder:Path = Path("autoencoder") / "models", filename:str = "autoencoder"):
        """
        Save the model state dictionary to a file.
        Args:
            folder (Path): Directory to save the model.
            filename (str): Name of the file to save the model.
        """
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), folder/f"{filename}.pt")

    
    def load(self, filepath: Path):
        self.load_state_dict(torch.load(filepath))