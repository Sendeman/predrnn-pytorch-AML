import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class AutoencoderModel(nn.Module):
    def __init__(self, latent_dim:int=100, epochs:int = 150, batch_size:int = 64, learning_rate:float = 1e-3):
        
        super(AutoencoderModel, self).__init__()
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.trained_epochs = 0

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 120x160 -> 60x80
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 60x80 -> 30x40
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 30x40 -> 15x20
            nn.ReLU(),
        )

        # Flatten and project to latent space
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(128 * 15 * 20, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 128 * 15 * 20)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 15x20 -> 30x40
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 30x40 -> 60x80
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # 60x80 -> 120x160
            nn.Sigmoid() # Images are normalized to [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        latent = self.fc_enc(self.flatten(encoded))
        decoded = self.fc_dec(latent).view(-1, 128, 15, 20)
        reconstructed = self.decoder(decoded)
        return reconstructed
    

    def save(self, folder:Path = None):
        # TODO: UGLY NAME BE SMARTER THAN THIS.
        if folder is None:
            folder = Path("autoencoder") / "models" 
        
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), folder/f"{self.latent_dim}_{self.trained_epochs}_{self.learning_rate}_{self.batch_size}_autoencoder.pt")
