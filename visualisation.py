#!/usr/bin/env python3

"""
Visualisation utilities for CIFAR-10 optimisation experiments.

Includes:
- Generic loss and accuracy curves (per iteration / epoch / step).
- Pareto front plotting for NSGA-II style multi-objective runs.
- Helpers for extracting and plotting NSGA-II logbook metrics.
- Comparison plots across multiple optimisation algorithms.

Safe to import from any optimisation script (SGD, SQN, DE, CSO, NSGA-II, etc.).
"""

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
    """
    Plot a single loss curve.

    Parameters
    ----------
    losses:
        Sequence of loss values, in temporal order.
    title:
        Title for the plot.
    xlabel:
        Label for the x-axis (e.g. 'Epoch', 'Step', 'Iteration').
    save_path:
        Optional path to save the figure as a PNG. If None, the plot is not saved.
    """
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
    """
    Plot a single accuracy curve (in percent).

    Parameters
    ----------
    accuracies:
        Sequence of accuracy values expressed in percent (0–100).
    title:
        Title for the plot.
    xlabel:
        Label for the x-axis (e.g. 'Epoch', 'Step', 'Generation').
    save_path:
        Optional path to save the figure as a PNG.
    """
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
    """
    Plot a Pareto front for a two-objective problem:
    - Objective 1: error (1 - accuracy) or validation loss.
    - Objective 2: Gaussian / L2 regulariser (sum of squares).

    Parameters
    ----------
    errors:
        Sequence of error values (e.g. 1 - accuracy, or loss).
    regs:
        Sequence of regulariser values (e.g. L2 norms).
    title:
        Title for the plot.
    save_path:
        Optional path to save the figure as a PNG.
    """
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
    """
    Extract per-generation metrics from a DEAP-style NSGA-II logbook.

    Assumes that:
    - logbook.select("min") returns a list of tuples per generation:
      (objective_1, objective_2), where
        objective_1 = error (1 - accuracy) OR validation loss
        objective_2 = L2 regulariser (sum of squares)
    - Accuracy is computed as (1 - error) * 100 if objective_1 is error.

    Parameters
    ----------
    logbook:
        A DEAP.tools.Logbook (or compatible) instance with a "min" field.

    Returns
    -------
    accuracies:
        List of accuracies per generation, in percent.
    errors:
        List of errors (1 - accuracy) per generation.
    regs:
        List of regulariser values per generation.
    """
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
    """
    Plot NSGA-II progress over generations:
    - Accuracy vs generation.
    - Error vs generation.
    - Regulariser vs generation.

    Parameters
    ----------
    logbook:
        DEAP-style Logbook with a "min" field.
    title_prefix:
        Prefix to use in figure titles (e.g. 'NSGA-II (accuracy-based)').
    save_prefix:
        If provided, three figures will be saved as:
        '{save_prefix}_accuracy.png',
        '{save_prefix}_error.png',
        '{save_prefix}_regulariser.png'.
    """
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
    """
    Plot loss curves from multiple optimisation methods on a single figure.

    Example
    -------
    plot_multiple_loss_curves(
        {
            "SGD": sgd_losses,
            "SQN": sqn_losses,
            "CSO": cso_best_losses,
            "DE": de_best_losses,
        },
        title="Loss Comparison Across Methods",
        xlabel="Step",
        save_path="loss_comparison.png",
    )

    Parameters
    ----------
    losses_by_label:
        Mapping from method name (legend label) to a sequence of loss values.
    title:
        Title for the plot.
    xlabel:
        Label for the x-axis.
    save_path:
        Optional path to save the figure as a PNG.
    """
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
