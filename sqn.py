#!/usr/bin/env python3

# https://arxiv.org/abs/1401.7020

import copy
from types import FunctionType
from collections.abc import Callable
from typing import Optional

import torch
import torchvision

import main
from main import CNN
import loss_function

base = torch.load("base.pt", weights_only=False)

training_data = torchvision.datasets.CIFAR10(root="./data",
                                             train=True,
                                             download=False,
                                             transform=loss_function.transform)


def get_batch(size: int) -> tuple:
    return next(iter(torch.utils.data.DataLoader(training_data, size, shuffle=True)))


def get_loss_of_model(model: main.CNN, batch_size: int) -> torch.Tensor:
    images, labels = get_batch(batch_size)
    device = next(model.parameters()).device
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    loss = loss_function.criterion(outputs, labels)
    return loss


def get_loss_of_params(params: torch.Tensor, batch_size: int) -> torch.Tensor:
    model = copy.deepcopy(base)
    model.to(params.device)
    model = loss_function.individual_to_model(params, model)
    return get_loss_of_model(model, batch_size)


def get_gradient_of_model(model: main.CNN, batch_size: int, for_hessian: bool = False) -> torch.Tensor:
    fc3_params = [p for p in model.fc3.parameters()]
    grads = torch.autograd.grad(get_loss_of_model(model, batch_size), fc3_params, create_graph=for_hessian)
    return torch.cat([g.flatten() for g in grads])


def get_gradient_of_params(params: torch.Tensor, batch_size: int, for_hessian: bool = False) -> torch.Tensor:
    model = copy.deepcopy(base)
    model.to(params.device)
    model = loss_function.individual_to_model(params, model)
    return get_gradient_of_model(model, batch_size, for_hessian=for_hessian)


def get_hessian_of_params(params: torch.Tensor, batch_size: int) -> torch.Tensor:
    model = copy.deepcopy(base)
    model.to(params.device)
    model = loss_function.individual_to_model(params, model)
    
    fc3_params = [p for p in model.fc3.parameters()]
    
    gradient = get_gradient_of_model(model, batch_size, for_hessian=True)
    
    hessian = torch.empty((len(gradient), len(gradient)), device=params.device)
    for i in range(len(hessian)):
        hessian[i] = torch.cat([g.flatten() for g in torch.autograd.grad(gradient[i], fc3_params, retain_graph=True, materialize_grads=True)])

    # print(gradient.sum().item(), hessian.sum().item())
    return hessian


def hessian_vector_product_of_params(params: torch.Tensor, multiplicand: torch.Tensor, batch_size: int) -> torch.Tensor:
    return get_hessian_of_params(params, batch_size)@multiplicand


def step_sequence(beta) -> FunctionType:
    def alpha(k):
        return beta/k

    return alpha


def sqn(params: torch.Tensor,
        num_epochs: int,
        max_memory: int,
        modulus: int,
        gradient: Callable[[torch.Tensor], torch.Tensor],
        hessian: Callable[[torch.Tensor], torch.Tensor],
        step_size: Callable,
        gradient_batch_size: int,
        hessian_batch_size: int,
        loss_batch_size: int = 1) -> torch.Tensor:
    assert len(params.shape) == 1
    
    device = params.device
    t = -1
    averaged_params = torch.zeros((num_epochs//modulus, len(params)), device=device)
    s = torch.empty_like(averaged_params)
    y = torch.empty_like(averaged_params)

    approximate_inverse_hessian = torch.empty((len(params), len(params)), device=device)
    
    for k in range(1, num_epochs+1):
        print(f"{k}\t{get_loss_of_params(params, loss_batch_size).item()}")
        averaged_params[t+1] += params
        
        if k <= 2*modulus:
            params -= step_size(k)*gradient(params, gradient_batch_size)
        else:
            approximate_inverse_hessian[:, :] = torch.eye(len(approximate_inverse_hessian), device=device)
            approximate_inverse_hessian *= torch.dot(s[t+1], y[t+1])/torch.dot(y[t+1], y[t+1])
            memory = min(t, max_memory)
            for j in range(t-memory+1, t+1):
                rho = 1/torch.dot(y[j+1], s[j+1])
                multiplier = torch.eye(len(params), device=device) - rho*torch.outer(s[j+1], y[j+1])
                approximate_inverse_hessian = multiplier@approximate_inverse_hessian
                approximate_inverse_hessian @= multiplier
                approximate_inverse_hessian += rho*torch.outer(s[j+1], s[j+1])

            params -= step_size(k)*approximate_inverse_hessian@gradient(params, gradient_batch_size)

        if k % modulus == 0:
            averaged_params[t+1] /= modulus
            t += 1  # this step is incorrectly placed at the top of the block in the paper
            if 0 < t < num_epochs//modulus-1:
                s[t+1] = averaged_params[t+1] - averaged_params[t]
                y[t+1] = hessian_vector_product_of_params(averaged_params[t+1], s[t+1], hessian_batch_size)

    return params


if __name__ == "__main__":
    print(base)
    params = loss_function.model_to_individual(base)
    print(params)
    
    trained_weights = sqn(params, 100, 10, 10, get_gradient_of_params, get_hessian_of_params, step_sequence(2), 50, 600)
    print(trained_weights)
    trained_model = loss_function.individual_to_model(trained_weights, base)
    print(trained_model)
    torch.save(trained_model, "sqn.pt")
