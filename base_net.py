import torch
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import sklearn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F


# class BasicNet(nn.Module):
#     # to have = [120, 84]
#     def __init__(self, input_size=784, output_size=10, layers=[10, 4]):
#         super().__init__()
#         self.d1 = nn.Linear(input_size, layers[0])  # hidden Layer 1
#         self.d2 = nn.Linear(layers[0], layers[1])
#         self.d3 = nn.Linear(layers[1], output_size)  # output Layer

#         self.layers = [self.d1, self.d2, self.d3]

#     def forward(self, X):
#         # print(X.shape)
#         X = F.relu(self.layers[0](X))
#         # print(X.shape)
#         X = F.relu(self.layers[1](X))
#         # print(X.shape)
#         X = self.layers[2](X)
#         # print(X.shape)
#         return F.log_softmax(X, dim=1)

class BasicNet(nn.Module):
    # to have = [120, 84]
    def __init__(self, input_size=784, output_size=10, layers=[5]):
        super().__init__()
        self.d1 = nn.Linear(input_size, layers[0])
        self.d2 = nn.Linear(layers[0], output_size)

        self.layers = [self.d1, self.d2]

    def forward(self, X):
        # print(X.shape)
        X = F.relu(self.layers[0](X))
        # print(X.shape)
        X = self.layers[1](X)
        # print(X.shape)
        return F.log_softmax(X, dim=1)

    # def get_layers(self):
    #     return [self.d1, self.d2, self.d3]


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    basic_net = BasicNet()
    # basic_net.to(device)
    print(basic_net)
