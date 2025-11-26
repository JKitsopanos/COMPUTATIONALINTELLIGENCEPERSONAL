#!/usr/bin/env python3

"""
Competitive Swarm Optimisation (CSO) on the final layer (fc3) of the CNN.
"""

from typing import Tuple, Callable

import sys
import torch

import main
from main import CNN
from loss_function import (
    function as make_loss_function,
    model_to_individual,
    individual_to_model,
)

sys.modules["__main__"].CNN = main.CNN

POPULATION_SIZE = 10


def initialise_population(base: CNN, population_size: int) -> torch.Tensor:
    """
    Creating an initial population of particles.
    Each particle is a flattened 1D vector representing all parameters
    """
    prototype = model_to_individual(base)
    num_params = prototype.numel()

    #Shape: (population_size, num_params)
    population = 2.0 * torch.rand(population_size, num_params) - 1.0
    return population


def initialise_velocities(population: torch.Tensor) -> torch.Tensor:
    """
    Initialising particle velocities with small random values.
    """
    return torch.randn_like(population) * 0.1


def evaluate_population(
    population: torch.Tensor,
    f: Callable[[torch.Tensor], float],
) -> torch.Tensor:
    """
    Evaluating each particle using the provided loss function f.
    """
    losses = []
    with torch.no_grad():
        for individual in population:
            individual = individual.flatten()
            loss_value = f(individual)
            losses.append(loss_value)

    return torch.tensor(losses, dtype=torch.float32)


def cso_update(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    losses: torch.Tensor,
    inertia_weight: float,
    winner_coeff: float,
    mean_coeff: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performing one CSO update.
    """
    num_particles, dim = positions.shape

    mean_position = positions.mean(dim=0)

    new_positions = positions.clone()
    new_velocities = velocities.clone()

    indices = torch.randperm(num_particles)

    for k in range(0, num_particles - 1, 2):
        idx1 = indices[k].item()
        idx2 = indices[k + 1].item()

        #Identify winner (lower loss) and loser (higher loss)
        if losses[idx1] <= losses[idx2]:
            winner_idx, loser_idx = idx1, idx2
        else:
            winner_idx, loser_idx = idx2, idx1

        r1 = torch.rand(dim)
        r2 = torch.rand(dim)

        #Current state of loser and winner
        loser_pos = positions[loser_idx]
        loser_vel = velocities[loser_idx]
        winner_pos = positions[winner_idx]

        #CSO loser velocity + position update
        updated_vel = (
            inertia_weight * loser_vel
            + winner_coeff * r1 * (winner_pos - loser_pos)
            + mean_coeff * r2 * (mean_position - loser_pos)
        )

        updated_pos = loser_pos + updated_vel
        updated_pos = torch.clamp(updated_pos, -1.0, 1.0)

        new_velocities[loser_idx] = updated_vel
        new_positions[loser_idx] = updated_pos

    return new_positions, new_velocities


def train(
    base: CNN,
    inertia_weight: float,
    winner_coeff: float,
    mean_coeff: float,
    population_size: int,
    steps: int,
) -> CNN:
    
    loss_f = make_loss_function(base)

    #Initialisation
    positions = initialise_population(base, population_size)
    velocities = initialise_velocities(positions)

    current_losses = evaluate_population(positions, loss_f)

    #Tracking global best solution for logging and final model
    best_index = torch.argmin(current_losses)
    global_best_position = positions[best_index].clone()
    global_best_loss = float(current_losses[best_index].item())

    print(f"Initial best loss: {global_best_loss:.4f}")

    #CSO iterations
    for step in range(steps):
        #Update swarm via CSO using current losses
        positions, velocities = cso_update(
            positions=positions,
            velocities=velocities,
            losses=current_losses,
            inertia_weight=inertia_weight,
            winner_coeff=winner_coeff,
            mean_coeff=mean_coeff,
        )

        current_losses = evaluate_population(positions, loss_f)

        best_index = torch.argmin(current_losses)
        candidate_best_loss = float(current_losses[best_index].item())
        if candidate_best_loss < global_best_loss:
            global_best_loss = candidate_best_loss
            global_best_position = positions[best_index].clone()

        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step + 1}/{steps} - best loss: {global_best_loss:.4f}")

    best_model = individual_to_model(global_best_position, base)
    return best_model


base = torch.load("base.pt", weights_only=False)

if __name__ == "__main__":
    best_model = train(
        base=base,
        inertia_weight=0.4,   
        winner_coeff=1.5,    
        mean_coeff=1.5,       
        population_size=POPULATION_SIZE,
        steps=50,
    )
    print("Finished CSO. Best model fc3 parameters updated.")
