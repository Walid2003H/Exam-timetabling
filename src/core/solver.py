"""Solver interface, optimization budget management, and execution results.

Implements the Fair Budget Rule (FEE units), standard result containers,
and the abstract base class for all optimization algorithms in the project.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time

import numpy as np

from src.core.instance import Instance


class Budget:
    """Manages evaluation-based budgets across algorithms using Full Evaluation Equivalents (FEE)."""

    def __init__(self, max_evals: float, max_seconds: float | None = None) -> None:
        """Initialize the budget counter."""
        self.max_evals = float(max_evals)
        self.max_seconds = float(max_seconds) if max_seconds is not None else None
        self._used_evals = 0.0
        self._start_time = time.perf_counter()

    def charge_full(self, n: int = 1) -> None:
        """Charge full evaluation equivalents (FEE = 1.0 per full solution evaluation)."""
        self._used_evals += float(n)

    def charge_delta(self, inst: Instance, i: int) -> None:
        """Charge incremental move evaluation cost: deg(i) / |E|."""
        if inst.n_edges <= 0:
            self._used_evals += 1.0 / float(max(1, inst.n_exams))
        else:
            deg = float(inst.degree(i))
            self._used_evals += deg / float(inst.n_edges)

    @property
    def used(self) -> float:
        """Return total consumed budget in FEE units."""
        return self._used_evals

    @property
    def exhausted(self) -> bool:
        """Check whether the evaluation or wall-clock budget has been exhausted."""
        if self._used_evals >= self.max_evals:
            return True
        if self.max_seconds is not None:
            if (time.perf_counter() - self._start_time) >= self.max_seconds:
                return True
        return False


@dataclass
class Result:
    """Container holding the final output and historical tracking of a solver run."""

    best_x: np.ndarray
    best_f: float
    feasible: bool
    history: np.ndarray  # Columns: [evals_used, iteration, best_f_so_far, seconds, diversity]
    elapsed: float
    evals: float
    hparams: dict[str, Any] = field(default_factory=dict)


class Solver(ABC):
    """Abstract base class for all optimization solvers (GA, PSO, ACO, Baselines)."""

    name: str = "base_solver"

    def __init__(self, **hparams: Any) -> None:
        """Initialize solver with algorithm-specific hyperparameters."""
        self.hparams = hparams

    @abstractmethod
    def solve(self, inst: Instance, budget: Budget, seed: int) -> Result:
        """Execute the optimization algorithm on a problem instance."""
        pass