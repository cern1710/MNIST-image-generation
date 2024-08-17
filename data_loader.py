from torchvision import transforms
from torch.utils.data import random_split
import torch.utils.data as data
import torchvision
import torch

def discretize(x, num_values=16):
    """Hard-coded discretization method into 4-bit representation."""
    return (x * num_values).long().clamp_(max=num_values-1)

def load_mnist(root='./data/', batch_size=128, num_workers=4, download=True):
    """Data loaders for 4-bit MNIST dataset."""
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(discretize)
    ])

    dataset = torchvision.datasets.MNIST(root, train=True,
                                         transform=data_transforms, download=download)
    test_dataset = torchvision.datasets.MNIST(root, train=False,
                                              transform=data_transforms, download=download)
    train_dataset, val_dataset = random_split(dataset, lengths=[54000, 6000],
                                              generator=torch.Generator().manual_seed(42))

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, drop_last=False)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, drop_last=False)

    return train_loader, val_loader, test_loader