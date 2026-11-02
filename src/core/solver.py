"""Fair budget manager (Budget), unified Result dataclass, and Solver interface (Section 3.2)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from src.core.instance import Instance


class Budget:
    """Tracks Full Evaluation Equivalent (FEE) budget usage and optional wall-clock timeout."""

    def __init__(self, max_evals: float, max_seconds: float | None = None) -> None:
        """Initialize budget with max_evals FEE limit and optional max_seconds limit."""
        self.max_evals = float(max_evals)
        self.max_seconds = float(max_seconds) if max_seconds is not None else None
        self._used = 0.0
        self._t0 = time.perf_counter()

    def charge_full(self, n: int = 1) -> None:
        """Charge n full solution evaluations (1 FEE each)."""
        self._used += float(n)

    def charge_delta(self, inst: Instance, i: int) -> None:
        """Charge a delta_move evaluation for exam i: adds deg(i) / |E| FEE."""
        self._used += float(inst.degree(i)) / float(max(1, inst.n_edges))

    @property
    def used(self) -> float:
        """Return total FEE used so far."""
        return self._used

    @property
    def elapsed(self) -> float:
        """Return elapsed wall-clock seconds since Budget creation."""
        return time.perf_counter() - self._t0

    @property
    def exhausted(self) -> bool:
        """Return True if FEE budget or wall-clock limit is reached."""
        if self._used >= self.max_evals:
            return True
        if self.max_seconds is not None and self.elapsed >= self.max_seconds:
            return True
        return False


@dataclass
class Result:
    """Unified solver output container (Section 3.2)."""

    best_x: np.ndarray
    best_f: float
    feasible: bool
    history: np.ndarray  # rows: [evals_used, iteration, best_f_so_far, seconds, diversity]
    elapsed: float
    evals: float
    hparams: dict[str, Any] = field(default_factory=dict)


class Solver(ABC):
    """Abstract base class for all timetabling solvers."""

    name: str = "solver"

    def __init__(self, **hparams: Any) -> None:
        """Store solver hyperparameters from YAML config."""
        self.hparams: dict[str, Any] = dict(hparams)

    @abstractmethod
    def solve(self, inst: Instance, budget: Budget, seed: int) -> Result:
        """Solve instance within budget using seed and return Result."""
        raise NotImplementedError
