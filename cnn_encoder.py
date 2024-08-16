import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self,
                 num_input_channels: int = 1,
                 num_filters: int = 32,
                 z_dim: int = 20):
        super().__init__()

        self.conv1 = nn.Conv2d(num_input_channels, num_filters, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        conv_output_size = num_filters * 28

        self.fc_mean = nn.Linear(conv_output_size, z_dim)
        self.fc_log_std = nn.Linear(conv_output_size, z_dim)

    def forward(self, x):
        x = x.float() / 15 * 2.0 - 1.0  # Normalize images in [-1, 1]

        # Apply conv layers with ReLU
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Flatten output from conv layers
        x = x.view(x.size(0), -1)

        # Compute the mean and log_std
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)

        return mean, log_std