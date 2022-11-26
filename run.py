import torchvision
from cmpnsgd import NSGD
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.functional as F

def train_single_step(model, Data, optimizer, c, lo):
    
    for batch in Data:  
        x, y = batch
        optimizer.zero_grad()
        out  = c(model(x))
        loss = lo(out, y)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    model = torchvision.models.resnet18(pretrained=False)
    optim = NSGD(model.parameters(), lr=0.1)
    L = nn.Softmax(dim=1)
    lo = nn.CrossEntropyLoss()
    
    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    for ep in range(100):
        train_single_step(model, train_dataloader, optim, L, lo)