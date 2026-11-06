"""Optimization progress and history tracker.

Periodically records fitness, diversity, runtime, and evaluation metrics
at fixed budget intervals (every 0.5% of total budget).
"""

from __future__ import annotations

import time

import numpy as np

from src.core.solver import Budget


class Tracker:
    """Tracks optimization trajectory and records history snapshots."""

    def __init__(self, budget: Budget, step_fraction: float = 0.005) -> None:
        """Initialize tracker with a budget and sampling step fraction.

        Args:
            budget: Shared Budget tracker instance.
            step_fraction: Fraction of budget between history snapshots (default 0.5%).
        """
        self.budget = budget
        self.step_fraction = float(step_fraction)
        self.max_evals = budget.max_evals
        self.interval = max(1.0, self.max_evals * self.step_fraction)
        self._start_time = time.perf_counter()

        self._next_checkpoint = 0.0
        self._history_records: list[list[float]] = []

    def record(
        self,
        iteration: int,
        best_f_so_far: float,
        diversity: float,
        force: bool = False,
    ) -> None:
        """Record a history snapshot if interval reached or forced.

        Args:
            iteration: Current algorithm iteration index.
            best_f_so_far: Best objective value discovered so far.
            diversity: Population diversity score.
            force: If True, forces recording regardless of budget interval.
        """
        evals_used = self.budget.used
        elapsed = time.perf_counter() - self._start_time

        if force or evals_used >= self._next_checkpoint or evals_used >= self.max_evals:
            self._history_records.append(
                [
                    float(evals_used),
                    float(iteration),
                    float(best_f_so_far),
                    float(elapsed),
                    float(diversity),
                ]
            )
            self._next_checkpoint = evals_used + self.interval

    def to_array(self) -> np.ndarray:
        """Return history records as a 2D numpy array of shape (num_snapshots, 5)."""
        if not self._history_records:
            return np.zeros((0, 5), dtype=np.float64)
        return np.array(self._history_records, dtype=np.float64)
