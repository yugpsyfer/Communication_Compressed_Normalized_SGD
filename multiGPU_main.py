import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score
from multiGPU_cmpnsgd import CNSGD
import wandb
from torch.distributed import ReduceOp
from resnet import resnet


def allocate_process_to_GPU(local_rank, args):
     
    world_rank =  local_rank % args['world_size']
    p=4
    q=20 
    beta=0.9

    if world_rank == 0:
        wandb.init(
          project="Communication Compressed INSGD",
          entity="yugansh",
          name="MultiGPU-CMPNSGD-one_bit",
          config={"p":p,"q":q,"beta":beta,"Rho":args['world_size']}
      ) 
    
    dist.init_process_group(backend=args["backend"],
    init_method=args["init_method"],
    group_name=args["name"],
    rank=world_rank,
    world_size=args["world_size"])

    torch.cuda.set_device(world_rank)
  
    trainLoader = args['trainLoader']
    testLoader = args['testLoader']


    model, criterion = prepare_model(world_rank)
    model.to(device='cuda')
    optimizer = CNSGD(model.parameters(),
     p=p, 
     q=q, 
     beta=beta,
     rho=args["world_size"], 
     compression_method='one_bit')
    
    for epoch in range(args["epochs"]):
        l = train_single_epoch(model, trainLoader, criterion, optimizer)
        # print("WORLD RANK: {} LOSS: {}".format(world_rank,l))

        if epoch%5 == 0: 
            model = average_models(model, world_size=args["world_size"])
            
            if world_rank == 0:
                accu_train = evaluate(model=model, testLoader=trainLoader)
                accu_test = evaluate(model=model, testLoader=testLoader)

                update_wandb_logs(loss=l,  
                accu_train=accu_train, 
                accu_test=accu_test)

                dist.barrier()
            else:
                dist.barrier()


def update_wandb_logs(loss, accu_train, accu_test):

    # print("Train Accuracy: {} || Test Accuracy: {} || LOSS: {}".format(accu_train, accu_test, loss))
    
    wandb.log({"Training Accuracy": accu_train, "Test Accracy": accu_test, "Loss":loss})


def average_models(model, world_size):

    for module in model.parameters():
        communication_tensor = module.data.clone()
        dist.all_reduce(communication_tensor, op=ReduceOp.SUM)
        communication_tensor.div_(world_size)
        module.data = communication_tensor.clone()    

    return model

    

def train_single_epoch(model, trainLoader, criterion, optimizer):
    batch_count = 0
    total_loss = 0

    for batch in trainLoader: 
        X, y = batch
        X = X.to('cuda')
        y = y.to('cuda')
        model.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss+=loss.detach()
        batch_count+=1

    return total_loss/batch_count


@torch.no_grad()
def evaluate(model, testLoader):
    b=0
    acc=0

    for batch in testLoader: 
        x, y = batch
        x = x.to('cuda')
        out  = model(x)
        out = torch.argmax(input=out, dim=-1)
        acc+=accuracy_score(y, out.cpu())
        b+=1
    
    return acc/b


def prepare_data():
    """
    Not implemented as DDP
    """
    print("Preparing data")

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


def prepare_model(rank):
    print("Preparing model for process on GPU: ",rank)
    outputfn = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    
    model = resnet('cifar10', 20)
    # model.fc = nn.Linear(512,10)  #10 classes in CIFAR10
    model.add_module(name="ouputFN", module=outputfn)
    
    return model, criterion


def run():
    nprocs = torch.cuda.device_count()
    trainLoader, testLoader =  prepare_data()
    barrier = mp.Barrier(nprocs)
  

    args=[{
        "world_size":nprocs,
        "backend": "nccl",
        "init_method": "tcp://127.0.0.1:21456",
        "name": "Compressed",
        "epochs":100,
        "trainLoader":trainLoader,
        "testLoader":testLoader,
        "barrier": barrier

    }]
    
    mp.spawn(fn=allocate_process_to_GPU, args=args, join=True, nprocs=nprocs)


if __name__=='__main__':
    run()