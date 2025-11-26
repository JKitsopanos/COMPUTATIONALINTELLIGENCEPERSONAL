#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from typing import Callable, Tuple


VAL_N: int = 2048
VAL_SEED: int = 0
BATCH_SIZE: int = 128
NUM_WORKERS: int = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


def make_val_loader(n: int = VAL_N, seed: int = VAL_SEED) -> DataLoader:
    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(testset), generator=g)[:n].tolist()
    return DataLoader(
        Subset(testset, idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

val_loader: DataLoader = make_val_loader()


def freeze_except_fc3(model: nn.Module) -> nn.Module:
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc3")
    return model



def model_to_individual(model: nn.Module, grad: bool = False) -> torch.Tensor:
    if grad:
        params = [p.flatten() for p in model.fc3.parameters()]
    else:
        params = [p.detach().flatten() for p in model.fc3.parameters()]
    if not params:
        raise ValueError("model.fc3 has no parameters to optimise.")
    return torch.cat(params)

def individual_to_model(individual: torch.Tensor, model: nn.Module) -> nn.Module:
    with torch.no_grad():
        idx = 0
        for p in model.fc3.parameters():
            num = p.numel()
            segment = individual[idx:idx + num].reshape_as(p)
            p.copy_(segment.to(device=p.device, dtype=p.dtype))
            idx += num
    
    return model


@torch.no_grad()
def eval_loss_and_acc(model: nn.Module) -> Tuple[float, float]:
    assert next(model.parameters()).device == device, "Model not on correct device."
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def _loss_function(model: nn.Module) -> float:
    loss_val, _ = eval_loss_and_acc(model)
    return loss_val

@torch.no_grad()
def _accuracy_function(model: nn.Module) -> float:
    _, acc = eval_loss_and_acc(model)
    return acc


def loss_function(individual: torch.Tensor, base_model: nn.Module) -> float:
    ind = individual.detach()

 
    ref = next(base_model.fc3.parameters())
    ind = ind.to(device=ref.device, dtype=ref.dtype)


    expected = sum(p.numel() for p in base_model.fc3.parameters())
    assert ind.numel() == expected, (
        f"Individual length {ind.numel()} != expected {expected}"
    )


    saved = [p.detach().clone() for p in base_model.fc3.parameters()]

    try:
        individual_to_model(ind, base_model)
        return _loss_function(base_model)
    finally:
        
        with torch.no_grad():
            for p, s in zip(base_model.fc3.parameters(), saved):
                p.copy_(s)

def accuracy_function(individual: torch.Tensor, base_model: nn.Module) -> float:
    ind = individual.detach()
    ref = next(base_model.fc3.parameters())
    ind = ind.to(device=ref.device, dtype=ref.dtype)

    expected = sum(p.numel() for p in base_model.fc3.parameters())
    assert ind.numel() == expected

    saved = [p.detach().clone() for p in base_model.fc3.parameters()]

    try:
        individual_to_model(ind, base_model)
        return _accuracy_function(base_model)
    finally:
        with torch.no_grad():
            for p, s in zip(base_model.fc3.parameters(), saved):
                p.copy_(s)

def function(base_model: nn.Module) -> Callable[[torch.Tensor], float]:

    base_model.to(device)
    freeze_except_fc3(base_model)

    def f(individual: torch.Tensor) -> float:
        return loss_function(individual, base_model)

    return f


def l2_regulariser_fc3(individual: torch.Tensor) -> float:
    ind = individual.detach().float()
    return torch.sum(ind * ind).item()

def nsga2_objectives(
        individual: torch.Tensor,
        base_model: nn.Module,
        use_accuracy: bool = False
) -> Tuple[float, float]:
    if use_accuracy:
        acc = accuracy_function(individual, base_model)
        obj1 = 1.0 - acc
    else:
        obj1 = loss_function(individual, base_model)

    obj2 = l2_regulariser_fc3(individual)
    return obj1, obj2
