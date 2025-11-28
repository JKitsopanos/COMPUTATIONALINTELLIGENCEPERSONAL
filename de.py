#!/usr/bin/env python3

"""Differential evolution with the original mutation scheme and binomial crossover (DE/rand/1/bin)."""

import random
from collections.abc import Iterable

import torch

import main
import loss_function as lf  # terrible naming hack

from main import CNN

POPULATION_SIZE = 4

base_model = torch.load("base.pt", weights_only=False)
loss_function = lf.function(base_model)


def generate_samples(model: main.CNN, n: int) -> torch.Tensor:
    assert n >= 4  # otherwise differential evolution cannot work

    # I retract the below comment.
    # This currently samples from a uniform distribution on [0,1); this should be replaced.
    return torch.rand((n, *lf.model_to_individual(model).shape))


def mutate(original: torch.Tensor, mutant: torch.Tensor, differential_weight) -> None:
    """Mutate the population in-place per the original scheme.

    Arguments:
    population -- a tensor where each row is an individual
    mutant -- a tensor to be overwritten
    differential_weight -- called F in the slides
    """

    assert original.shape == mutant.shape
    
    n = len(original)
    assert n >= 4

    for i in range(n):
        t, r, s = random.sample([j for j in range(n) if j != i], 3)
        mutant[i] = original[t] + differential_weight*(original[r] - original[s])


def binomial_crossover(original: torch.Tensor, mutant: torch.Tensor, cr) -> None:
    """Modify mutant in-place to be the offspring produced by binomial crossover.

    Arguments:
    original
    mutant
    cr -- crossover rate, a.k.a. crossover probability
    """

    assert original.shape == mutant.shape
    assert 0 <= cr <= 1

    n, d = original.shape

    # There's probably a vectorised way to do this.
    for i in range(n):
        for j in range(d):
            if random.random() >= cr:
                mutant[i][j] = original[i][j]


def select_survivors(original: torch.Tensor, offspring: torch.Tensor, f) -> None:
    assert original.shape == offspring.shape

    n = len(original)

    for i in range(n):
        if f(offspring[i]) <= f(original[i]):
            original[i] = offspring[i]


def get_best(population: Iterable[torch.Tensor]) -> torch.Tensor:
    return min(population, key=loss_function)


def train(base: main.CNN,
          differential_weight,
          cr,
          population_size: int,
          steps: int,
          loss_modulus: int = 1) -> main.CNN:
    assert 0 <= cr <= 1
    assert population_size >= 4
    
    population = generate_samples(base, population_size)
    mutant = torch.empty_like(population)

    for i in range(steps):
        print(i, end="\t")
        if i % loss_modulus == 0:
            print(loss_function(get_best(population)))
        else:
            print()
            
        mutate(population, mutant, differential_weight)
        binomial_crossover(population, mutant, cr)
        select_survivors(population, mutant, loss_function)

    return lf.individual_to_model(get_best(population), base)


base = torch.load("base.pt", weights_only=False)

if __name__ == "__main__":
    print(base)
    print(train(base, 0.5, 0.5, POPULATION_SIZE, 1))
