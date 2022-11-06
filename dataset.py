from torchvision import datasets, transforms
import sklearn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F

import hyperparams as hyperparams

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(
    root="../Datasets", train=True, download=False, transform=transform)

test_dataset = datasets.MNIST(
    root="../Datasets", train=False, download=False, transform=transform)

train_loader = DataLoader(
    train_dataset, batch_size=hyperparams.batch_size, shuffle=False)
test_loader = DataLoader(
    test_dataset, batch_size=hyperparams.batch_size, shuffle=False)
