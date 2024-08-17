import torch.nn as nn

class CNNDecoder(nn.Module):
    def __init__(self,
                 num_input_channels: int = 16,
                 num_filters: int = 32,
                 z_dim: int = 20):
        super().__init__()

        self.fc = nn.Linear(z_dim, num_filters * 98)
        self.deconv1 = nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(num_filters, num_input_channels, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, z):
        x = self.fc(z)  # Forward latent vector through FC layer
        x = x.view(z.size(0), -1, 7, 7) # Match dims for deconv

        # Apply transposed conv layers
        x = self.relu(self.deconv1(x))
        x = self.deconv2(x)

        return x

    @property
    def device(self):
        return next(self.parameters()).device