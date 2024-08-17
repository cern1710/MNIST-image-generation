import torch
import numpy as np

Tensor = torch.Tensor

def sample_reparameterize(mean: Tensor, std: Tensor) -> Tensor:
    """Sample reparameterize trick to sample from a distribution."""
    return mean + std * torch.randn_like(mean)

def KL_divergence(mean, log_std):
    """KL divergence of given distributions to unit Gaussians over last dim."""
    std = torch.exp(log_std)
    return -0.5 * torch.sum(1 + log_std - mean**2 - std**2, dim=-1)

def convert_ELBO_to_BPD(elbo, input_shape):
    """Converts summed NLL from ELBO to bits per dimension score."""
    return elbo / (np.prod(input_shape[1:]) * np.log(2))