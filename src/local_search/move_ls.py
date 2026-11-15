"""Single-exam relocate (move) local search operator.

Provides first-improvement and best-improvement move passes that strictly
account for incremental evaluations via `Budget.charge_delta` and use an
isolated `np.random.Generator`.
"""

from __future__ import annotations

import numpy as np

from src.core.instance import Instance
from src.core.problem import delta_move
from src.core.solver import Budget

_EPS = 1e-9


def move_ls(
    inst: Instance,
    x: np.ndarray,
    budget: Budget | None = None,
    rng: np.random.Generator | None = None,
    max_passes: int = 20,
    first_improvement: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Improve timetable `x` by relocating single exams to better time slots.

    A move is accepted if it strictly reduces hard conflict weight (`delta_V < 0`),
    or if it keeps `delta_V == 0` while strictly reducing Carter proximity cost
    (`delta_f < -1e-9`). Every tested move charges `deg(i) / |E|` FEE when a
    `Budget` object is provided, and terminates immediately if `budget.exhausted`.

    Args:
        inst: Problem instance in CSR format.
        x: Starting timetable of shape (N,).
        budget: Optional FEE budget tracker; if None, moves are not charged.
        rng: Isolated random generator (created with default seed 0 if None).
        max_passes: Maximum full sweeps over all exams.
        first_improvement: Apply first improving move immediately when True;
            otherwise scan all slots for an exam and pick the best slot.

    Returns:
        Tuple `(improved_x, total_delta_f, total_delta_v)`.

    Time Complexity:
        O(max_passes * T * |E|) in the worst case, with each candidate slot
        evaluated in O(deg(i)) via `delta_move`.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    current_x = np.array(x, dtype=np.int32, copy=True)
    total_df = 0.0
    total_dv = 0.0

    if inst.n_slots <= 1 or max_passes <= 0:
        return current_x, total_df, total_dv

    for _ in range(max_passes):
        if budget is not None and budget.exhausted:
            break

        improved_in_pass = False
        # Permute exam order each pass to avoid index bias
        exam_order = rng.permutation(inst.n_exams)

        for exam_raw in exam_order:
            if budget is not None and budget.exhausted:
                break

            exam_i = int(exam_raw)
            # Isolated exams have zero edges and cannot change f(x) or V(x)
            if inst.degree(exam_i) == 0:
                continue

            cur_slot = int(current_x[exam_i])
            slot_order = rng.permutation(inst.n_slots)

            best_slot = cur_slot
            best_df = 0.0
            best_dv = 0.0

            for slot_raw in slot_order:
                if budget is not None and budget.exhausted:
                    break

                cand_slot = int(slot_raw)
                if cand_slot == cur_slot:
                    continue

                if budget is not None and inst.n_edges > 0:
                    move_cost = float(inst.degree(exam_i)) / float(inst.n_edges)
                    if not budget.can_afford(move_cost):
                        break

                df, dv = delta_move(inst, current_x, exam_i, cand_slot)
                if budget is not None:
                    budget.charge_delta(inst, exam_i)

                # Accept move if it reduces hard violations or reduces soft cost at zero delta_V
                is_better = (dv < best_dv - _EPS) or (
                    abs(dv - best_dv) <= _EPS and df < best_df - _EPS
                )
                if is_better:
                    best_slot = cand_slot
                    best_df = df
                    best_dv = dv
                    if first_improvement:
                        break

            if best_slot != cur_slot:
                current_x[exam_i] = best_slot
                total_df += best_df
                total_dv += best_dv
                improved_in_pass = True

        if not improved_in_pass:
            break

    return current_x, total_df, total_dv
