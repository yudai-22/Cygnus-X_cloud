import torch.nn as nn


class Conv3dAutoencoder(nn.Module):
    def __init__(self, latent):
        super(Conv3dAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),  # output=(60, 56, 56)
            nn.ReLU(True),
            nn.Conv3d(64, 32, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),  # output=(30, 28, 28)
            nn.ReLU(True),
            nn.Conv3d(32, 16, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),  # output=(15, 14, 14)
            nn.ReLU(True),
            nn.Conv3d(16, 16, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(0, 0, 0)),  #output=(6, 6, 6)
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(16 * 6 * 6 * 6, latent),  # Adjust the size based on the flattened output
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent, 16 * 6 * 6 * 6),
            nn.Unflatten(1, (16, 6, 6, 6)),# Unflatten back to (32, 5, 25, 25)
            nn.ConvTranspose3d(
                16, 16, kernel_size=(6, 6, 6), stride=(2, 2, 2), padding=(1, 1, 1)
            ),  # output=(8, 56, 56)
            nn.ReLU(True),
            nn.ConvTranspose3d(
                16, 32, kernel_size=(5, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)
            ),  # output=(8, 56, 56)
            nn.ReLU(True),
            nn.ConvTranspose3d(
                32, 64, kernel_size=(5, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)
            ),  # output=(10, 111, 111)
            nn.ReLU(True),
            nn.ConvTranspose3d(
                64, 32, kernel_size=(4, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
            ),  # output=(12, 112, 112)
            nn.ReLU(True),
            nn.ConvTranspose3d(
                32, 1, kernel_size=(3, 2, 2), stride=(1, 1, 1), padding=(0, 0, 0)
            ),  # output=(8, 56, 56),
            nn.Sigmoid(),  # to scale output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
