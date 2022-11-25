import torch.functional as F
import torchvision
from cmpnsgd import NSGD
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def train_single_step(model, Data, optimizer):
    
    for batch in Data:  
        x, y = batch
        optimizer.zero_grad()
        loss  = model(x)
        
        loss.backward()
        optimizer.step()



if __name__ == 'main':
    model = torchvision.models.resnet18(pretrained=False)
    optim = NSGD(model.parameters, lr=0.1)
    print(model)
    9/0
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    for ep in range(100):
        train_single_step(model, train_dataloader, optim)