import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


criterion = nn.CrossEntropyLoss()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


def make_val_loader(n=2048, seed=0):
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(testset), generator=g)[:n].tolist()
    return DataLoader(
        Subset(testset, idx),
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

val_loader = make_val_loader()



def model_to_individual(model):
   
    params = []
    for p in model.fc3.parameters():
        params.append(p.detach().cpu().flatten())
    return torch.cat(params)


def individual_to_model(individual, model):
 
    with torch.no_grad():
        idx = 0
        for p in model.fc3.parameters():
            num = p.numel()
            p.copy_(individual[idx:idx+num].reshape(p.shape))
            idx += num
    return model




@torch.no_grad()
def _loss_function(model):
    
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_items = 0

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        total_items += y.size(0)

    return total_loss / total_items


def loss_function(individual, base_model):
   
    ind = individual.detach().float()
    updated = individual_to_model(ind, base_model)
    return _loss_function(updated)
