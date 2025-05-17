import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoencoderModel(nn.Module):
    def __init__(self, latent_dim=100):
        super(AutoencoderModel, self).__init__()
        self.latent_dim = latent_dim

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
        print("prior:", x.shape)
        encoded = self.encoder(x)
        print("encoded:", encoded.shape)
        latent = self.fc_enc(self.flatten(encoded))
        print("latent:", latent.shape)
        decoded = self.fc_dec(latent).view(-1, 128, 15, 20)
        reconstructed = self.decoder(decoded)
        print("reconstructed:", reconstructed.shape)
        return reconstructed
