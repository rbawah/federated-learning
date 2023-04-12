import torch
import torch.nn as nn
from torch.nn import Conv2d, Linear
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
# import tqdm as tqdm

class GlobalModel(torch.nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.conv1 = Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = Linear(2048, 128)
        self.fc2 = Linear(128, 10)

    def forward(self, data):
        data = self.conv1(data)
        data = F.relu(data)
        data = self.conv2(data)
        data = F.relu(x)
        data = self.conv3(data)
        data = F.relu(data)
        data = torch.flatten(data, 1)
        data = self.fc1(data)
        data = F.relu(data)
        data = self.fc2(data)
        out = F.log_softmax(x, dim=1)

        return out