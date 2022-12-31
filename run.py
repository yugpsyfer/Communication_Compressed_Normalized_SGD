import torchvision
from cmpnsgd import NSGD
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from insgd import INSGD
from torch.optim import Adam, SGD, Adadelta
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import wandb


def train_single_step(model, Data, optimizer, c, lo, device):
    Lp = 0
    count=0
    for batch in Data: 
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
        out  = c(model(x))
        loss = lo(out, y)
        Lp+=loss.detach()
        loss.backward()
        optimizer.step()
        count+=1

    return Lp/count


@torch.no_grad()
def evaluate(model, Data, criterion, device):
    b=0
    acc=0

    for batch in Data: 
        x, y = batch
        x = x.to(device)
        out  = criterion(model(x))
        out = torch.argmax(input=out, dim=-1)
        acc+=accuracy_score(y, out.cpu())
        b+=1
    
    return acc/b

def plot(vals):
    epochs = [i for i in range(1, 101)]
    
    axes= plt.subplots(2)
    
    axes[0].plot(epochs, vals['train'])
    axes[0].set_ylim(1)
    axes[0].set_xlim(100)
    axes[0].set_title('Train vs Epoch')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_ylabel('Epochs')
    
    axes[1].plot(epochs, vals['test'])
    axes[1].set_ylim(1)
    axes[1].set_xlim(100)
    axes[1].set_title('Test vs Epoch')
    axes[1].set_xlabel('Accuracy')
    axes[1].set_ylabel('Epochs')
    plt.show()

if __name__ == '__main__':
    
    L = nn.Softmax(dim=1)
    lo = nn.CrossEntropyLoss()
    
    device = torch.device('cuda')
  
    
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

    """TRAINING WITH INSGD"""
    
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(512,10)  #10 classes in CIFAR10
    model.to(device)
    optim = INSGD(model.parameters(), lr=0.1)
    
    wandb.init(
        project="Communication Compressed INSGD",
        entity="yugansh",
        name="INSGD",
        config={"p":1,"q":10,"beta":0.9}
    )    

    print("Training started Ingd")

    for ep in range(100):
        LOSS = train_single_step(model, train_dataloader, optim, L, lo, device)
        accu_train = evaluate(model, train_dataloader, L, device)
        accu_test = evaluate(model, test_dataloader, L, device)
        wandb.log({"Training Accuracy": accu_train,
                        "Test Accracy": accu_test})   


    """TRAINING WITH ADAM"""

    wandb.init(
        project="Communication Compressed INSGD",
        entity="yugansh",
        name="ADAM",
        config={"lr":0.0001,"weight_decay":1e-6,"eps":1e-3}
    )  

    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(512,10)  #10 classes in CIFAR10
    model.to(device)
    optim = Adam(model.parameters(), lr=0.0001, weight_decay=1e-6, eps=1e-3)


    print("Training started for ADAM")
    for ep in range(100):
        LOSS = train_single_step(model, train_dataloader, optim, L, lo, device)
        accu_train = evaluate(model, train_dataloader, L, device)
        accu_test = evaluate(model, test_dataloader, L, device)
        
        wandb.log({"Training Accuracy": accu_train,
                        "Test Accracy": accu_test})


    """TRAINING WITH SGD"""
    wandb.init(
        project="Communication Compressed INSGD",
        entity="yugansh",
        name="SGD_mom",
        config={"lr":0.1,"weight_decay":5*1e-4,"momentum":0.9}
    ) 

    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(512,10)  #10 classes in CIFAR10
    model.to(device)
    optim = SGD(model.parameters(), lr=0.1, weight_decay=5*1e-4, momentum=0.9)

    print("Training started for SGD")
    for ep in range(100):
        LOSS = train_single_step(model, train_dataloader, optim, L, lo, device)
        accu_train = evaluate(model, train_dataloader, L, device)
        accu_test = evaluate(model, test_dataloader, L, device)
        wandb.log({"Training Accuracy": accu_train,
                        "Test Accracy": accu_test}) 
    
