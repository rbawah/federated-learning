
def main():

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import random_split, DataLoader
    from torchvision import datasets, transforms
    import numpy as np
    from tqdm import tqdm

    from models.global_model import GlobalModel

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

    server = GlobalModel().to(device)
    clients = [GlobalModel().to(device) for i in range(clients_selected)]

    for model in clients:
        model.load_state_dict(server.state_dict())

    # Try a different optimizer
    client_optimizers = [optim.SGD(client.parameters(), lr=0.1) for client in clients]

    for round in range(num_rounds):
        




