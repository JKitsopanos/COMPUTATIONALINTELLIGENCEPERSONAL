import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch

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
    return DataLoader(Subset(testset, idx),
                      batch_size=128, shuffle=False, num_workers=2)

val_loader = make_val_loader()

@torch.no_grad()
def _loss_function(model: CNN) -> float:
    model.eval()
    total_loss, total = 0.0, 0

    for x, y in val_loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs

    return total_loss / total  

def loss_function(individual: torch.Tensor) -> float:
    candidate = individual_to_model(individual, base)
    return _loss_function(candidate)
