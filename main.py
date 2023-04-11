
def main():

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import random_split, DataLoader
    from torchvision import datasets, transforms
    import numpy as np
    from tqdm import tqdm

    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

    print(f'Device: {device}')
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    num_clients = 100
    clients_selected = 10
    num_rounds = 5
    epochs = 5
    batch_size = 32

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_split = random_split(train_set, [int(train_set.data.shape[0]/num_clients) for i in range(num_clients)])
    train_loader = [DataLoader(data, batch_size=batch_size, shuffle=True) for data in train_split]
    test_loader = [DataLoader(test_set, batch_size=batch_size, shuffle=True)]



