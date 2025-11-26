#!/usr/bin/env python3

"""
Particle Swarm Optimisation (PSO) on the final layer (fc3) of the CNN.

Design:
- The CNN is first trained with SGD in main.py and saved as base.pt.
- All layers except fc3 are frozen.
- Optimising ONLY fc3 using PSO, treating all its parameters (weights + bias) as a single 1D vector.

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

# torch.save(network, "base.pt") in main.py pickled the class as __main__.CNN.
# Here we tell Python that __main__.CNN actually refers to main.CNN so that
# torch.load can unpickle the model.
sys.modules["__main__"].CNN = main.CNN


POPULATION_SIZE = 10


def initialise_population(base: CNN, population_size: int) -> torch.Tensor:
    prototype = model_to_individual(base)
    num_params = prototype.numel()
    population = 2.0 * torch.rand(population_size, num_params) - 1.0
    return population


def initialise_velocities(population: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(population) * 0.1


def evaluate_population(
    population: torch.Tensor,
    f: Callable[[torch.Tensor], float],
) -> torch.Tensor:
    """
    Evaluate each particle using the provided loss function f.
    """
    losses = []
    with torch.no_grad():
        for individual in population:
            individual = individual.flatten()
            loss_value = f(individual)
            losses.append(loss_value)

    return torch.tensor(losses, dtype=torch.float32)


def pso_update(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    personal_best_positions: torch.Tensor,
    global_best_position: torch.Tensor,
    inertia_weight: float,
    cognitive_coeff: float,
    social_coeff: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform one PSO velocity + position update.
    """
    r1 = torch.rand_like(positions)
    r2 = torch.rand_like(positions)

    cognitive_term = cognitive_coeff * r1 * (personal_best_positions - positions)
    social_term = social_coeff * r2 * (global_best_position - positions)

    new_velocities = inertia_weight * velocities + cognitive_term + social_term
    new_positions = positions + new_velocities

    return new_positions, new_velocities


def train(
    base: CNN,
    inertia_weight: float,
    cognitive_coeff: float,
    social_coeff: float,
    population_size: int,
    steps: int,
) -> CNN:
    """
    Main PSO loop.
    Returns:
        CNN model whose fc3 has been replaced with the best-found parameters.
    """
    loss_f = make_loss_function(base)

    # --- Initialisation ---
    positions = initialise_population(base, population_size)
    velocities = initialise_velocities(positions)

    current_losses = evaluate_population(positions, loss_f)

    #Personal bests for each particle
    personal_best_positions = positions.clone()
    personal_best_losses = current_losses.clone()

    #Global best across swarm
    best_index = torch.argmin(current_losses)
    global_best_position = positions[best_index].clone()
    global_best_loss = float(current_losses[best_index].item())

    print(f"Initial best loss: {global_best_loss:.4f}")

    # --- PSO iterations ---
    for step in range(steps):
        positions, velocities = pso_update(
            positions,
            velocities,
            personal_best_positions,
            global_best_position,
            inertia_weight,
            cognitive_coeff,
            social_coeff,
        )

        current_losses = evaluate_population(positions, loss_f)

        #Update personal bests where improved
        improved_mask = current_losses < personal_best_losses
        personal_best_positions[improved_mask] = positions[improved_mask]
        personal_best_losses[improved_mask] = current_losses[improved_mask]

        #Update global best
        best_index = torch.argmin(personal_best_losses)
        candidate_best_loss = float(personal_best_losses[best_index].item())
        if candidate_best_loss < global_best_loss:
            global_best_loss = candidate_best_loss
            global_best_position = personal_best_positions[best_index].clone()

        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step + 1}/{steps} - best loss: {global_best_loss:.4f}")

    #---Put best parameters back into the model ---
    best_model = individual_to_model(global_best_position, base)
    return best_model


#Loading the frozen model
base = torch.load("base.pt", weights_only=False)


if __name__ == "__main__":
    best_model = train(
        base=base,
        inertia_weight=0.7,
        cognitive_coeff=1.5,
        social_coeff=1.5,
        population_size=POPULATION_SIZE,
        steps=50,
    )
    print("Finished PSO. Best model fc3 parameters updated.")
