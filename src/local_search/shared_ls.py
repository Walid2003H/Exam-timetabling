"""Unified local search entry points shared across P4, P5 (GA), P6 (PSO), and P7 (ACO).

Provides `intensify` (combining single-exam moves, Kempe chains, and swaps) and
`local_search_first_improvement` with strict `rng` isolation and `Budget` FEE
accounting.
"""

from __future__ import annotations

import numpy as np

from src.core.instance import Instance
from src.core.solver import Budget
from src.local_search.kempe_ls import kempe_ls
from src.local_search.move_ls import move_ls
from src.local_search.swap_ls import swap_ls


def local_search_first_improvement(
    inst: Instance,
    x: np.ndarray,
    max_iter: int = 20,
    budget: Budget | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run first-improvement single-exam move search on `x`.

    Maintains backward compatibility with callers passing `(inst, x, max_iter)`
    while supporting explicit `budget` and `rng` arguments required by the
    interface contract.

    Args:
        inst: Problem instance in CSR format.
        x: Starting timetable array of shape (N,).
        max_iter: Maximum full passes over exams.
        budget: Optional FEE budget tracker.
        rng: Isolated NumPy random generator.

    Returns:
        Improved timetable array of shape (N,) and dtype int32.
    """
    improved_x, _, _ = move_ls(
        inst,
        x,
        budget=budget,
        rng=rng,
        max_passes=max_iter,
        first_improvement=True,
    )
    return improved_x


def intensify(
    inst: Instance,
    x: np.ndarray,
    budget: Budget,
    rng: np.random.Generator,
    operator: str = "hybrid",
    max_passes: int = 5,
    kempe_steps: int = 30,
    swap_trials: int = 50,
) -> np.ndarray:
    """Memetic local search helper for GA, PSO, and ACO hybrid solvers.

    Applies the chosen neighborhood operator (`"move"`, `"kempe"`, `"swap"`, or
    `"hybrid"`) while strictly respecting `budget.exhausted` and charging every
    incremental move via `budget.charge_delta`.

    Args:
        inst: Problem instance in CSR format.
        x: Candidate timetable of shape (N,).
        budget: Shared FEE budget tracker.
        rng: Isolated random generator.
        operator: One of `"move"`, `"kempe"`, `"swap"`, or `"hybrid"`.
        max_passes: Maximum sweeps for `move_ls`.
        kempe_steps: Number of Kempe chain trials when `operator` is `"kempe"` or `"hybrid"`.
        swap_trials: Number of swap trials when `operator` is `"swap"` or `"hybrid"`.

    Returns:
        Locally refined timetable array of shape (N,).
    """
    current = np.array(x, dtype=np.int32, copy=True)
    if budget.exhausted:
        return current

    op = operator.lower().strip()
    if op == "move":
        current, _, _ = move_ls(inst, current, budget=budget, rng=rng, max_passes=max_passes)
    elif op == "kempe":
        current, _, _ = kempe_ls(inst, current, budget=budget, rng=rng, max_steps=kempe_steps)
    elif op == "swap":
        current, _, _ = swap_ls(inst, current, budget=budget, rng=rng, max_trials=swap_trials)
    elif op == "hybrid":
        current, _, _ = move_ls(inst, current, budget=budget, rng=rng, max_passes=max_passes)
        if not budget.exhausted and kempe_steps > 0:
            current, _, _ = kempe_ls(inst, current, budget=budget, rng=rng, max_steps=kempe_steps)
        if not budget.exhausted and swap_trials > 0:
            current, _, _ = swap_ls(inst, current, budget=budget, rng=rng, max_trials=swap_trials)
        if not budget.exhausted:
            current, _, _ = move_ls(inst, current, budget=budget, rng=rng, max_passes=1)
    else:
        raise ValueError(
            f"Unknown operator {operator!r}; expected 'move', 'kempe', 'swap', or 'hybrid'."
        )

    return current
