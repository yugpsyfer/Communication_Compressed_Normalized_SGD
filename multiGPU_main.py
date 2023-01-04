import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from sklearn.metrics import accuracy_score
from multiGPU_cmpnsgd import CNSGD


def allocate_process_to_GPU(local_rank, args):
    world_rank = args["world_size"] + local_rank

    dist.init_process_group(backend=args["backend"],
    init_method=args["init_method"],
    group_name=args["name"],
    rank=world_rank,
    world_size=args["world_size"])

    torch.cuda.set_device(world_rank)

    trainLoader, testLoader =  prepare_data()
    model, criterion = prepare_model()
    optimizer = CNSGD(model.parameters())

    for epoch in range(args["epochs"]):
        l = train_single_epoch(model, trainLoader, criterion, optimizer)
        print("LOSS: ",l)


def train_single_epoch(model, trainLoader, criterion, optimizer):
    batch_count = 0
    total_loss = 0

    for batch in trainLoader: 
        
        X, y = batch
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss+=loss.detach()
        batch_count+=1
    
    return total_loss/batch_count


@torch.no_grad()
def evaluate():
    pass


def prepare_data():
    """
    Not implemented as DDP
    """
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

    return train_dataloader, test_dataloader


def prepare_model():
    outputfn = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(512,10)  #10 classes in CIFAR10
    model.add_module(name="ouputFN", module=outputfn)
    
    return model, criterion


def run():
    nprocs = torch.cuda.device_count()
    args={
        "world_size":nprocs,
        "backend": "nccl",
        "init_method": "tcp://127.0.0.1:21456",
        "name": "Compressed",
        "epochs":10
    }
    
    mp.spawn(fn=allocate_process_to_GPU, args=args, join=True, nprocs=nprocs)

    
