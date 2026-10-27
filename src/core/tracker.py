"""Optimization progress and history tracker.

Periodically records fitness, diversity, runtime, and evaluation metrics
at fixed budget intervals (every 0.5% of total budget).
"""

from __future__ import annotations

import time

import numpy as np

from src.core.instance import Instance
from src.core.problem import diversity, fitness, is_feasible


class Tracker:
    """Tracks optimization trajectory and records history snapshots."""

    def __init__(self, inst: Instance, budget_max: float, sample_interval_ratio: float = 0.005) -> None:
        """Initialize tracker."""
        self.inst = inst
        self.budget_max = float(budget_max)
        self.interval = max(1.0, self.budget_max * float(sample_interval_ratio))
        self._start_time = time.perf_counter()

        self.best_f = float("inf")
        self.best_x: np.ndarray | None = None
        self.best_feasible = False

        self._next_checkpoint = 0.0
        self._history_records: list[list[float]] = []

    def update(
        self,
        evals_used: float,
        iteration: int,
        population_or_solution: np.ndarray,
    ) -> None:
        """Evaluate current population or solution and record snapshot if interval reached."""
        arr = np.asarray(population_or_solution)
        if arr.ndim == 1:
            pop = arr.reshape(1, -1)
        elif arr.ndim == 2:
            pop = arr
        else:
            raise ValueError("Tracker update expects 1D or 2D NumPy array.")

        for x in pop:
            if is_feasible(self.inst, x):
                f_val = fitness(self.inst, x, lam=0.0)
                if f_val < self.best_f:
                    self.best_f = f_val
                    self.best_x = np.array(x, dtype=np.int32)
                    self.best_feasible = True

        if evals_used >= self._next_checkpoint or evals_used >= self.budget_max:
            elapsed = time.perf_counter() - self._start_time
            div_val = diversity(pop) if pop.shape[0] > 1 else 0.0
            recorded_f = self.best_f if self.best_feasible else float("inf")

            self._history_records.append([
                float(evals_used),
                float(iteration),
                float(recorded_f),
                float(elapsed),
                float(div_val),
            ])
            self._next_checkpoint = evals_used + self.interval

    def get_history(self) -> np.ndarray:
        """Return history records as a 2D numpy array of shape (num_snapshots, 5)."""
        if not self._history_records:
            return np.zeros((0, 5), dtype=np.float64)
        return np.array(self._history_records, dtype=np.float64)