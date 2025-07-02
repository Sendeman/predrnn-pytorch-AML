import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class AutoencoderModel(nn.Module):
    def __init__(self, epochs:int=150, encoder:nn.Sequential=None, decoder:nn.Sequential=None):
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
        
        self.encoder = encoder
        self.decoder = decoder


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

    
    def load(self, filepath: Path, device='cuda'):
        """
        Load the model state dictionary from a file.
        Args:
            filepath (Path): Path to the file containing the model state dictionary.
            device: The device to load the model onto
        """
        self.load_state_dict(torch.load(filepath, map_location=device))


architectures = {
    1: {"encoder" : nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 1x128x128-> 32x128x128
            nn.LeakyReLU(),

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), # 32x128x128 -> 16x128x128
            nn.LeakyReLU(),

            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), # 16x128x128 -> 16x64x64
            nn.LeakyReLU(),

            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1), # 16x64x64 -> 1x64x64
        ),
        "decoder" : nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=3, stride=1, padding=1), # 1x64x64 -> 16x64x64 
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # 16x64x64 -> 16x128x128
            nn.LeakyReLU(),

            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1), # 16x128x128 -> 32x128x128
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1), # 32x128x128 -> 1x128x128
            nn.Sigmoid()
        )
        },
    

    2: {"encoder" : nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 1x128x128 -> 32x128x128
            nn.LeakyReLU(),

            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1), # 32x128x128 -> 16x64x64
            nn.LeakyReLU(),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), # 16x64x64-> 16x64x64
            nn.LeakyReLU(),

            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1), # 16x64x64 -> 8x32x32
            nn.LeakyReLU(),

            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1), # 8x32x32-> 1x32x32
        ),
        "decoder" : nn.Sequential(
            nn.ConvTranspose2d(1, 8, kernel_size=3, stride=1, padding=1), # 1x32x32-> 8x32x32
            nn.LeakyReLU(),

            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # 8x32x32 -> 16x64x64
            nn.LeakyReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1), # 16x64x64 -> 16x64x64
            nn.LeakyReLU(),

            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 16x64x64-> 32x128x128
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1), # 32x128x128 -> 1x128x128
            nn.Sigmoid()
        )
        },
    3: {"encoder" : nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 1x128x128 -> 32x128x128
            nn.LeakyReLU(),

            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1), # 32x128x128 -> 16x64x64
            nn.LeakyReLU(),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), # 16x64x64 -> 16x64x64
            nn.LeakyReLU(),

            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1), # 16x64x64 -> 8x32x32
            nn.LeakyReLU(),

            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1), # 8x32x32 -> 8x32x32
            nn.LeakyReLU(),

            nn.Conv2d(8, 1, kernel_size=3, stride=2, padding=1), # 8x32x32 -> 1x16x16
        ),
        "decoder" : nn.Sequential(
            nn.ConvTranspose2d(1, 8, kernel_size=3, stride=2, padding=1, output_padding=1), # 1x16x16 -> 8x32x32
            nn.LeakyReLU(),

            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=1, padding=1), # 8x32x32 -> 8x32x32
            nn.LeakyReLU(),

            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # 8x32x32 -> 16x64x64
            nn.LeakyReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1), # 16x64x64 -> 16x64x64
            nn.LeakyReLU(),

            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 16x64x64 -> 32x128x128
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1), # 32x128x128-> 32x128x128
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1), # 32x128x128 -> 1x128x128
            nn.Sigmoid()
        )
        }
    
}