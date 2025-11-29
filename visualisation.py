#!/usr/bin/env python3

from typing import Sequence, Tuple, List

import matplotlib.pyplot as plt


def plot_loss_curve(
    losses: Sequence[float],
    title: str,
    xlabel: str = "Iteration",
    save_path: str | None = None,
) -> None:
    
    steps = range(1, len(losses) + 1)

    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_accuracy_curve(
    accuracies: Sequence[float],
    title: str,
    xlabel: str = "Iteration",
    save_path: str | None = None,
) -> None:
    
    steps = range(1, len(accuracies) + 1)

    plt.figure()
    plt.plot(steps, accuracies)
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_pareto_front(
    errors: Sequence[float],
    regs: Sequence[float],
    title: str,
    save_path: str | None = None,
) -> None:
   
    plt.figure()
    plt.scatter(errors, regs)
    plt.xlabel("Error (1 - accuracy)")
    plt.ylabel("Gaussian regulariser (sum of squares)")
    plt.title(title)
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def extract_nsga_logbook_curves(logbook) -> Tuple[List[float], List[float], List[float]]:
  
    mins = logbook.select("min")  # list of (error, reg) per generation

    errors = [m[0] for m in mins]
    regs = [m[1] for m in mins]
    accuracies = [(1.0 - e) * 100.0 for e in errors]

    return accuracies, errors, regs
