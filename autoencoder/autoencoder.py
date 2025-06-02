import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class AutoencoderModel(nn.Module):
    def __init__(self, latent_dim:int=100, epochs:int = 150):
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
        

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), # 1x120x160 -> 32 x 120 x 160
            nn.LeakyReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=4, padding=1), # 32x120x160 -> 64x30x40
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 64x30x40 -> 128x15x20
            nn.LeakyReLU(),

            nn.Flatten(),
            nn.Linear(128 * 20 * 15, latent_dim)
        )

        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 20 * 15),
            nn.Unflatten(1, (128, 15, 20)),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128x15x20 -> 64x30x40
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x30x40 -> 32x60x80
            nn.LeakyReLU(),

            nn.Conv2d (32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d (32, 1, kernel_size=4, stride=2, padding=1),  # 32x60x80-> 1x120x160
            nn.Sigmoid(),
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
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


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