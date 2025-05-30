import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class AutoencoderModel(nn.Module):
    def __init__(self, latent_dim:int=100, epochs:int = 150, a_function=nn.ReLU()):
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
        self.trained_epochs = 0
        self.activation = a_function

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), # 1x120x160 -> 64 x 120 x 160
            self.activation,

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 64x120x160 -> 128x60x80
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.activation,

            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1), # 128x60x80 -> 192x30x40
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.activation,

            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1), # 192x30x40 -> 256x15x20
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.activation
        )

        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(256 * 20 * 15, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 20 * 15)
        self.unflatten = nn.Unflatten(1, (256, 15, 20))



        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(256, 192, kernel_size=3, stride=1, padding=1),  # 32x15x20 -> 16x30x40
            self.activation,

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(192, 128, kernel_size=3, stride=1, padding=1),  # 16x30x40 -> 16x60x80
            self.activation,

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # 16x60x80 -> 8x120x160
            self.activation,

            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),  # 4x120x160 -> 1x120x160
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
        flattened = self.flatten(encoded)
        latent = self.fc_enc(flattened)
        decode = self.fc_dec(latent)
        unflattened = self.unflatten(decode)
        reconstructed = self.decoder(encoded)
        return reconstructed
    

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
        """
        Load the model state dictionary from a file.
        Args:
            filepath (Path): Path to the file containing the model state dictionary.
        """
        self.load_state_dict(torch.load(filepath))