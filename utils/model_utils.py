import torch
import torch.nn.functional as F

# try different loss criterion
def client_update(client_model, optimizer, train_loader, epochs=5, device=None):
    client_model.train()
    for epoch in range(epochs):
        for idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            out = client_model(data)
            loss = F.nll_loss(out, label)
            loss.backward()
            optimizer.step()

    return loss.item()

def global_aggregate(server, clients):
    server_dict = server.state_dict()
    for ii in server_dict.keys():
        server_dict[ii] = torch.stack([clients[jj].state_dict()[ii] for jj in range(len(clients))], 0).mean(0)
    server.load_state_dict(server_dict)
    for model in clients:
        model.load_state_dict(server.state_dict())


def evaluate(server, test_loader, device=None):
    server.eval()
    loss, corrects = 0, 0
    with torch.no_grad():
        for idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output = server(data)
            loss += F.nll_loss(output, label, reduction='sum').item()
            preds = output.argmax(dim=1, keepdim=True)
            corrects += preds.eq(label.view_as(preds)).sum().item()

    loss /= len(test_loader.dataset)
    accuracy = corrects / len(test_loader.dataset)
    return loss, accuracy

