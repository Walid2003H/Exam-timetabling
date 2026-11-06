"""Solver interface, optimization budget management, and execution results.

Implements the Fair Budget Rule (FEE units), standard result containers,
and the abstract base class for all optimization algorithms in the project.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.core.instance import Instance


class Budget:
    """Manage evaluation budgets across solvers using Full Evaluation Equivalents (FEE)."""

    def __init__(self, max_evals: float, max_seconds: float | None = None) -> None:
        """Initialize the budget counter.

        Args:
            max_evals: Maximum allowed evaluations in FEE units.
            max_seconds: Optional wall-clock time limit in seconds.

        Time Complexity:
            O(1).
        """
        if max_evals <= 0:
            raise ValueError(f"max_evals must be positive, got {max_evals}.")
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError(f"max_seconds must be positive, got {max_seconds}.")

        self.max_evals = float(max_evals)
        self.max_seconds = float(max_seconds) if max_seconds is not None else None
        self._used_evals = 0.0
        self._start_time = time.perf_counter()

    def charge_full(self, n: int = 1) -> None:
        """Charge `n` full evaluation equivalents (1.0 FEE per full evaluation).

        Args:
            n: Number of full evaluations to charge.

        Time Complexity:
            O(1).
        """
        if n < 0:
            raise ValueError(f"Charged evaluations must be non-negative, got {n}.")
        self._used_evals += float(n)

    def charge_delta(self, inst: Instance, i: int) -> None:
        """Charge incremental move evaluation cost equal to deg(i) / |E|.

        When |E| == 0, charges 0.0 to avoid division by zero.

        Args:
            inst: Problem instance in CSR format.
            i: Exam index whose move was evaluated incrementally.

        Time Complexity:
            O(1).
        """
        if inst.n_edges <= 0:
            return
        self._used_evals += float(inst.degree(i)) / float(inst.n_edges)

    @property
    def used(self) -> float:
        """Return total consumed budget in FEE units."""
        return self._used_evals

    @property
    def exhausted(self) -> bool:
        """Return True if either FEE budget or wall-clock limit has been reached."""
        if self._used_evals >= self.max_evals:
            return True
        if self.max_seconds is not None:
            if (time.perf_counter() - self._start_time) >= self.max_seconds:
                return True
        return False


@dataclass
class Result:
    """Standard container for solver output and convergence history.

    Attributes:
        best_x: Best timetable array of shape (N,) found during the run.
        best_f: Carter proximity cost f(x) of `best_x` (inf if infeasible).
        feasible: True if `best_x` satisfies all hard constraints.
        history: 2D array with rows [evals_used, iteration, best_f_so_far, seconds, diversity].
        elapsed: Total wall-clock runtime in seconds.
        evals: Total FEE budget consumed.
        hparams: Dictionary of solver hyperparameters used in the run.
    """

    best_x: np.ndarray
    best_f: float
    feasible: bool
    history: np.ndarray
    elapsed: float
    evals: float
    hparams: dict[str, Any] = field(default_factory=dict)


class Solver(ABC):
    """Abstract base class for all baseline and metaheuristic solvers."""

    name: str = "base_solver"

    def __init__(self, hparams: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize solver with hyperparameters passed as a dict or keyword args.

        Args:
            hparams: Optional dictionary of hyperparameter values.
            **kwargs: Additional hyperparameter key-value pairs.

        Time Complexity:
            O(K) where K is the number of hyperparameters.
        """
        merged: dict[str, Any] = {}
        if hparams is not None:
            merged.update(hparams)
        merged.update(kwargs)
        self.hparams = merged

    @abstractmethod
    def solve(self, inst: Instance, budget: Budget, seed: int) -> Result:
        """Run the solver on `inst` within `budget` using deterministic `seed`.

        Args:
            inst: Problem instance in CSR format.
            budget: Fair budget tracker instance.
            seed: Integer seed for `np.random.default_rng(seed)`.

        Returns:
            Result container with the best solution and convergence history.
        """
        raise NotImplementedError
