"""Convenience wrappers and solver registry for P4 baseline algorithms.

Re-exports `RandomSearchSolver`, `DsaturSolver`, `SimulatedAnnealingSolver`,
and `ExactBranchAndBoundSolver`, as well as functional helpers `random_search`
and `dsatur_heuristic`.
"""

from __future__ import annotations

import numpy as np

from src.baselines.dsatur import DsaturSolver, dsatur_construct
from src.baselines.exact_bb import ExactBranchAndBoundSolver
from src.baselines.random_search import RandomSearchSolver
from src.baselines.sa import SimulatedAnnealingSolver
from src.core.instance import Instance
from src.core.solver import Budget, Result, Solver

BASELINE_SOLVERS: dict[str, type[Solver]] = {
    "random": RandomSearchSolver,
    "dsatur": DsaturSolver,
    "sa": SimulatedAnnealingSolver,
    "exact": ExactBranchAndBoundSolver,
}


def get_baseline_solver(name: str, **hparams) -> Solver:
    """Instantiate a baseline solver by its canonical name (`random`, `dsatur`, `sa`, `exact`)."""
    key = name.strip().lower()
    if key not in BASELINE_SOLVERS:
        raise ValueError(
            f"Unknown baseline solver {name!r}; expected one of {sorted(BASELINE_SOLVERS)}."
        )
    return BASELINE_SOLVERS[key](**hparams)


def random_search(inst: Instance, budget: Budget, seed: int) -> Result:
    """Run `RandomSearchSolver` and return the unified `Result` object."""
    solver = RandomSearchSolver()
    return solver.solve(inst, budget, seed)


def dsatur_heuristic(inst: Instance, seed: int) -> np.ndarray:
    """Construct a single DSATUR timetable using an isolated RNG seeded with `seed`."""
    rng = np.random.default_rng(seed)
    return dsatur_construct(inst, rng, slot_selection="min_proximity", use_repair=True)
