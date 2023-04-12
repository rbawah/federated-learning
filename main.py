
def main():

    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import random_split, DataLoader, Subset
    from torchvision import datasets, transforms
    import numpy as np
    # from tqdm import tqdm

    from models.global_model import GlobalModel
    from utils.model_utils import client_update, global_aggregate, evaluate

    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

    print(f'Device: {device}')
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    num_clients = 100
    num_selected = 10
    num_rounds = 5
    epochs = 5
    batch_size = 32

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    # Load and create datasets for i.i.d (homogenous) scenario
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_split = random_split(train_set, [int(train_set.data.shape[0]/num_clients) for i in range(num_clients)])
    train_loader = [DataLoader(data, batch_size=batch_size, shuffle=True) for data in train_split]
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    server = GlobalModel().to(device)
    clients = [GlobalModel().to(device) for i in range(num_selected)]

    for model in clients:
        model.load_state_dict(server.state_dict())

    # Try a different optimizer
    client_optimizers = [optim.SGD(client.parameters(), lr=0.1) for client in clients]

    for round in range(num_rounds):
        client_idx = np.random.permutation(num_clients)[:num_selected]

        loss = 0
        for ii in range(num_selected):
            loss += client_update(clients[ii], client_optimizers[ii], train_loader[client_idx[ii]], epochs=epochs, device=device)

        global_aggregate(server, clients)
        test_loss, test_accuracy = evaluate(server, test_loader, device=device)

        print(f'Round {round}')
        print(f'Avg Train Loss {(loss/num_selected):.3f} | Test Loss {test_loss:.3f} | Test Accuracy {test_accuracy:.3f}')

    # Load and create datasets for non-i.i.d scenario
    # Use same test_loader
    labels_niid = torch.stack([train_set.targets == i for i in range(10)])
    labels_split_niid = list()
    for i in range(num_selected):
        labels_split_niid += torch.split(torch.where(labels_niid[(2*i):(2*(i+1))].sum(0))[0], int(60000/num_clients))
    
    train_split = [Subset(train_set, ii) for ii in labels_split_niid]
    train_loader = [DataLoader(data, batch_size=batch_size, shuffle=True) for data in train_split]

    for round in range(num_rounds):
        client_idx = np.random.permutation(num_clients)[:num_selected]

        loss = 0
        for ii in range(num_selected):
            loss += client_update(clients[ii], client_optimizers[ii], train_loader, epochs=epochs, device=device)

        global_aggregate(server, clients)
        test_loss, test_accuracy = evaluate(server, test_loader, device=device)

        print(f'Round {round}')
        print(f'Avg Train Loss {(loss/num_selected):.3f} | Test Loss {test_loss:.3f} | Test Accuracy {test_accuracy:.3f}')


if __name__ == '__main__':
    main()