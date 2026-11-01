"""Pairwise slot-swap local search operator (`swap_ls`).

Evaluates swapping the time slots assigned to two distinct exams using two
sequential `delta_move` evaluations with exact rollback if rejected.
"""

from __future__ import annotations

import numpy as np

from src.core.instance import Instance
from src.core.problem import delta_move
from src.core.solver import Budget

_EPS = 1e-9


def evaluate_swap_delta(
    inst: Instance,
    x: np.ndarray,
    exam_i: int,
    exam_j: int,
    budget: Budget | None = None,
) -> tuple[float, float]:
    """Compute exact (delta_f, delta_V) for swapping slots of `exam_i` and `exam_j`.

    Decomposes the swap into two O(deg) single-exam moves by temporarily placing
    `exam_i` into `x[exam_j]`, querying `delta_move` for `exam_j`, and restoring
    `x[exam_i]`. This accurately accounts for any direct edge weight c_ij between
    `exam_i` and `exam_j` without scanning the whole graph.

    Args:
        inst: Problem instance in CSR format.
        x: Current timetable vector of shape (N,) (restored before return).
        exam_i: First exam index.
        exam_j: Second exam index.
        budget: Optional FEE budget tracker.

    Returns:
        Tuple `(delta_f, delta_V)` for the swap.

    Time Complexity:
        O(deg(exam_i) + deg(exam_j)).
    """
    s_i = int(x[exam_i])
    s_j = int(x[exam_j])
    if s_i == s_j or exam_i == exam_j:
        return 0.0, 0.0

    # Step 1: move exam_i from s_i to s_j
    df1, dv1 = delta_move(inst, x, exam_i, s_j)
    if budget is not None:
        budget.charge_delta(inst, exam_i)

    x[exam_i] = s_j
    # Step 2: move exam_j from s_j to s_i under updated slot of exam_i
    df2, dv2 = delta_move(inst, x, exam_j, s_i)
    if budget is not None:
        budget.charge_delta(inst, exam_j)

    # Restore original state of x
    x[exam_i] = s_i
    return df1 + df2, dv1 + dv2


def swap_ls(
    inst: Instance,
    x: np.ndarray,
    budget: Budget | None = None,
    rng: np.random.Generator | None = None,
    max_trials: int = 250,
) -> tuple[np.ndarray, float, float]:
    """Stochastic first-improvement pairwise swap local search.

    Samples pairs of exams assigned to different slots and swaps their slots
    whenever the swap reduces hard violations (`delta_V < 0`) or preserves
    feasibility (`delta_V == 0`) while reducing proximity cost (`delta_f < 0`).

    Args:
        inst: Problem instance in CSR format.
        x: Starting timetable of shape (N,).
        budget: Optional FEE budget tracker.
        rng: Isolated random generator.
        max_trials: Maximum candidate pairs to inspect.

    Returns:
        Tuple `(improved_x, total_delta_f, total_delta_v)`.

    Time Complexity:
        O(max_trials * avg_deg).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    current_x = np.array(x, dtype=np.int32, copy=True)
    total_df = 0.0
    total_dv = 0.0

    if inst.n_exams < 2 or inst.n_slots <= 1 or max_trials <= 0:
        return current_x, total_df, total_dv

    for _ in range(max_trials):
        if budget is not None and budget.exhausted:
            break

        exam_i = int(rng.integers(0, inst.n_exams))
        exam_j = int(rng.integers(0, inst.n_exams - 1))
        if exam_j >= exam_i:
            exam_j += 1

        if current_x[exam_i] == current_x[exam_j]:
            continue

        df, dv = evaluate_swap_delta(inst, current_x, exam_i, exam_j, budget=budget)
        if dv < -_EPS or (abs(dv) <= _EPS and df < -_EPS):
            s_i = int(current_x[exam_i])
            current_x[exam_i] = current_x[exam_j]
            current_x[exam_j] = s_i
            total_df += df
            total_dv += dv

    return current_x, total_df, total_dv
