from cnn_encoder import CNNEncoder
from cnn_decoder import CNNDecoder
import torch
import torch.nn.functional as F
from torch import nn, optim
from utils import *

class VAE(nn.Module):
    def __init__(self, num_filters, z_dim, learning_rate):
        """torch nn.Module describing all components to train our VAE."""
        super(VAE, self).__init__()
        self.encoder = CNNEncoder(z_dim=z_dim, num_filters=num_filters)
        self.decoder = CNNDecoder(z_dim=z_dim, num_filters=num_filters)
        self.learning_rate = learning_rate

    def forward(self, imgs):
        """Calculates VAE loss for given image batch."""
        # Encoding and reparameterization
        mean, log_std = self.encoder(imgs)
        z = sample_reparameterize(mean, torch.exp(log_std))

        # Decoding
        decoded_imgs = self.decoder(z)

        # Reconstruction loss
        L_rec = F.cross_entropy(decoded_imgs, imgs.squeeze(1),
                                reduction='none').sum([1, 2]).mean()

        # Regularization loss
        L_reg = KL_divergence(mean, log_std).mean()

        # Bits per dimension
        bpd = convert_ELBO_to_BPD(L_rec + L_reg, imgs.shape).mean()

        return L_rec, L_reg, bpd