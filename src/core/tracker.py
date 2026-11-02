"""Tracker utility logging best feasible objective every 0.5% of the FEE budget (Section 3.4)."""

import numpy as np

from src.core.solver import Budget


class Tracker:
    """Logs [evals_used, iteration, best_f_so_far, seconds, diversity] checkpoints."""

    def __init__(self, budget: Budget, step_fraction: float = 0.005) -> None:
        """Initialize tracker with periodic logging step (default 0.5% of max_evals)."""
        self.budget = budget
        self.step = max(1.0, budget.max_evals * step_fraction)
        self._next_eval = 0.0
        self._rows: list[list[float]] = []

    def record(
        self,
        iteration: int,
        best_f_so_far: float,
        diversity: float = 0.0,
        force: bool = False,
    ) -> None:
        """Record a convergence point when budget usage crosses the next step threshold."""
        used = self.budget.used
        if force or used >= self._next_eval:
            self._rows.append(
                [
                    float(used),
                    float(iteration),
                    float(best_f_so_far),
                    float(self.budget.elapsed),
                    float(diversity),
                ]
            )
            while self._next_eval <= used:
                self._next_eval += self.step

    def to_array(self) -> np.ndarray:
        """Return logged history as a 2D float64 NumPy array of shape (K, 5)."""
        if not self._rows:
            return np.zeros((0, 5), dtype=np.float64)
        return np.asarray(self._rows, dtype=np.float64)
