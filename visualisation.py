#!/usr/bin/env python3

from __future__ import annotations

from typing import Sequence, Tuple, List, Mapping

import matplotlib.pyplot as plt


def _finalise_plot(
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str | None,
    legend: bool = False,
) -> None:
    """
    Internal helper to apply common plot formatting and saving.
    """
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    if legend:
        plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_loss_curve(
    losses: Sequence[float],
    title: str,
    xlabel: str = "Iteration",
    save_path: str | None = None,
) -> None:

    if len(losses) == 0:
        raise ValueError("plot_loss_curve received an empty sequence of losses.")

    steps = range(1, len(losses) + 1)

    plt.figure()
    plt.plot(steps, losses)
    _finalise_plot(
        title=title,
        xlabel=xlabel,
        ylabel="Loss",
        save_path=save_path,
        legend=False,
    )


def plot_accuracy_curve(
    accuracies: Sequence[float],
    title: str,
    xlabel: str = "Iteration",
    save_path: str | None = None,
) -> None:
   
    if len(accuracies) == 0:
        raise ValueError("plot_accuracy_curve received an empty sequence of accuracies.")

    steps = range(1, len(accuracies) + 1)

    plt.figure()
    plt.plot(steps, accuracies)
    _finalise_plot(
        title=title,
        xlabel=xlabel,
        ylabel="Accuracy (%)",
        save_path=save_path,
        legend=False,
    )


def plot_pareto_front(
    errors: Sequence[float],
    regs: Sequence[float],
    title: str,
    save_path: str | None = None,
) -> None:
   
    if len(errors) == 0 or len(regs) == 0:
        raise ValueError("plot_pareto_front received empty sequences.")
    if len(errors) != len(regs):
        raise ValueError(
            f"plot_pareto_front expected errors and regs of same length, "
            f"got {len(errors)} and {len(regs)}."
        )

    plt.figure()
    plt.scatter(errors, regs)
    _finalise_plot(
        title=title,
        xlabel="Error (1 - accuracy)",
        ylabel="Gaussian regulariser (sum of squares)",
        save_path=save_path,
        legend=False,
    )


def extract_nsga_logbook_curves(logbook) -> Tuple[List[float], List[float], List[float]]:
   
    mins = logbook.select("min")  # list of (error, reg) per generation

    if len(mins) == 0:
        raise ValueError("extract_nsga_logbook_curves: logbook has no 'min' entries.")

    errors = [float(m[0]) for m in mins]
    regs = [float(m[1]) for m in mins]
    accuracies = [(1.0 - e) * 100.0 for e in errors]

    return accuracies, errors, regs


def plot_nsga_logbook_progress(
    logbook,
    title_prefix: str = "NSGA-II",
    save_prefix: str | None = None,
) -> None:
    
    accuracies, errors, regs = extract_nsga_logbook_curves(logbook)
    generations = range(1, len(accuracies) + 1)

    # Accuracy over generations
    plt.figure()
    plt.plot(generations, accuracies)
    _finalise_plot(
        title=f"{title_prefix}: Accuracy per Generation",
        xlabel="Generation",
        ylabel="Accuracy (%)",
        save_path=None if save_prefix is None else f"{save_prefix}_accuracy.png",
        legend=False,
    )

    # Error over generations
    plt.figure()
    plt.plot(generations, errors)
    _finalise_plot(
        title=f"{title_prefix}: Error per Generation",
        xlabel="Generation",
        ylabel="Error (1 - accuracy)",
        save_path=None if save_prefix is None else f"{save_prefix}_error.png",
        legend=False,
    )

    # Regulariser over generations
    plt.figure()
    plt.plot(generations, regs)
    _finalise_plot(
        title=f"{title_prefix}: Regulariser per Generation",
        xlabel="Generation",
        ylabel="Gaussian regulariser (sum of squares)",
        save_path=None if save_prefix is None else f"{save_prefix}_regulariser.png",
        legend=False,
    )


def plot_multiple_loss_curves(
    losses_by_label: Mapping[str, Sequence[float]],
    title: str,
    xlabel: str = "Iteration",
    save_path: str | None = None,
) -> None:
    
    if not losses_by_label:
        raise ValueError("plot_multiple_loss_curves received an empty mapping.")

    plt.figure()

    for label, losses in losses_by_label.items():
        if len(losses) == 0:
            continue
        steps = range(1, len(losses) + 1)
        plt.plot(steps, losses, label=label)

    _finalise_plot(
        title=title,
        xlabel=xlabel,
        ylabel="Loss",
        save_path=save_path,
        legend=True,
    )
