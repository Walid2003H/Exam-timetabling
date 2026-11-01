"""Saturation degree graph coloring heuristic (`dsatur`) based on Brelaz (1979).

Constructs timetables by iteratively selecting the unscheduled exam with the
largest number of distinct time slots already assigned to its conflicting
neighbours (saturation degree), breaking ties by static conflict degree and
random perturbation. Supports multi-start restarts with optional local search
refinement within the available FEE budget.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.core.instance import Instance
from src.core.problem import (
    diversity,
    proximity_cost,
    repair,
    violation_weight,
)
from src.core.solver import Budget, Result, Solver
from src.core.tracker import Tracker
from src.local_search.move_ls import move_ls


def dsatur_construct(
    inst: Instance,
    rng: np.random.Generator,
    slot_selection: str = "min_proximity",
    use_repair: bool = True,
) -> np.ndarray:
    """Construct a single timetable using the Brelaz (1979) DSATUR heuristic.

    At each step, picks the unscheduled exam maximizing:
      1. Saturation degree (number of distinct slots among already-scheduled neighbours).
      2. Static degree in the conflict graph.
      3. Uniform random tie-breaker from `rng`.

    When assigning a slot to the chosen exam, prefers conflict-free slots that
    minimize incremental Carter proximity cost against already-scheduled neighbours.

    Args:
        inst: Problem instance in CSR format.
        rng: Isolated NumPy random generator.
        slot_selection: `"min_proximity"` (greedy Carter cost) or `"random_feasible"`.
        use_repair: Run `repair` if a conflict could not be avoided during construction.

    Returns:
        Constructed timetable array of shape (N,) and dtype int32.

    Time Complexity:
        O(N^2 + |E| * T).
    """
    n_exams = inst.n_exams
    n_slots = inst.n_slots
    indptr = inst.indptr
    indices = inst.indices
    weights = inst.weights

    x = np.full(n_exams, -1, dtype=np.int32)
    unassigned = np.ones(n_exams, dtype=bool)
    degrees = np.diff(indptr)
    tie_noise = rng.random(n_exams)

    # Track which slots are used by neighbours of each exam: nbr_slot_used[i, s] = True
    nbr_slot_used = np.zeros((n_exams, n_slots), dtype=bool)
    sat_deg = np.zeros(n_exams, dtype=np.int32)

    for _ in range(n_exams):
        # Select unassigned exam with highest (sat_deg, static_degree, tie_noise)
        scores = np.where(unassigned, sat_deg * 1000000.0 + degrees * 100.0 + tie_noise, -1.0)
        u = int(np.argmax(scores))

        nbrs = indices[indptr[u] : indptr[u + 1]]
        wts = weights[indptr[u] : indptr[u + 1]]
        assigned_mask = ~unassigned[nbrs]
        used_nbr_slots = x[nbrs[assigned_mask]]
        used_nbr_wts = wts[assigned_mask]

        # Count conflict weight in each slot against already-assigned neighbours
        slot_conf = np.bincount(used_nbr_slots, weights=used_nbr_wts, minlength=n_slots)
        min_c = np.min(slot_conf)
        cand_slots = np.flatnonzero(slot_conf == min_c)

        if (
            cand_slots.shape[0] == 1
            or used_nbr_slots.shape[0] == 0
            or slot_selection == "random_feasible"
        ):
            chosen_slot = int(cand_slots[int(rng.integers(0, cand_slots.shape[0]))])
        else:
            # Among slots with minimum conflict, choose the one minimizing Carter proximity penalty
            diffs = np.abs(cand_slots[:, None] - used_nbr_slots[None, :])
            prox_mask = (diffs >= 1) & (diffs <= 5)
            prox_pen = np.where(
                prox_mask,
                (1 << np.clip(5 - diffs, 0, 5)) * used_nbr_wts[None, :],
                0,
            )
            prox_scores = np.sum(prox_pen, axis=1) + rng.random(cand_slots.shape[0]) * 1e-3
            chosen_slot = int(cand_slots[int(np.argmin(prox_scores))])

        x[u] = chosen_slot
        unassigned[u] = False

        # Update saturation degrees of unscheduled neighbours
        unassigned_nbrs = nbrs[unassigned[nbrs]]
        new_sat = unassigned_nbrs[~nbr_slot_used[unassigned_nbrs, chosen_slot]]
        nbr_slot_used[new_sat, chosen_slot] = True
        sat_deg[new_sat] += 1

    if use_repair and violation_weight(inst, x) > 0.0:
        x = repair(inst, x, rng, max_passes=6)

    return x


class DsaturSolver(Solver):
    """Multi-start DSATUR constructive solver (`dsatur`)."""

    name: str = "dsatur"

    def __init__(self, hparams: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize DsaturSolver.

        Supported hyperparameters:
            slot_selection (str): "min_proximity" or "random_feasible" (default "min_proximity").
            use_repair (bool): Apply conflict repair if needed (default True).
            polish_with_move_ls (bool): Apply move_ls after construction (default False).
            max_restarts (int): Maximum multi-start constructions (default 200).
        """
        defaults = {
            "slot_selection": "min_proximity",
            "use_repair": True,
            "polish_with_move_ls": False,
            "max_restarts": 200,
        }
        if hparams:
            defaults.update(hparams)
        defaults.update(kwargs)
        super().__init__(defaults)

    def solve(self, inst: Instance, budget: Budget, seed: int) -> Result:
        """Run multi-start DSATUR within `budget`.

        Args:
            inst: Problem instance in CSR format.
            budget: Fair evaluation budget tracker.
            seed: Deterministic seed for `np.random.default_rng(seed)`.

        Returns:
            Result with best feasible solution found.
        """
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        tracker = Tracker(budget)

        slot_sel = str(self.hparams.get("slot_selection", "min_proximity"))
        use_rep = bool(self.hparams.get("use_repair", True))
        use_ls = bool(self.hparams.get("polish_with_move_ls", False))
        max_restarts = max(1, int(self.hparams.get("max_restarts", 200)))

        best_feasible_x: np.ndarray | None = None
        best_feasible_f: float = float("inf")
        fallback_x = np.zeros(inst.n_exams, dtype=np.int32)
        best_infeasible_v = float("inf")

        recent_pool: list[np.ndarray] = []
        iteration = 0

        while not budget.exhausted and iteration < max_restarts:
            cand_x = dsatur_construct(inst, rng, slot_selection=slot_sel, use_repair=use_rep)
            v_val = violation_weight(inst, cand_x)
            f_val = proximity_cost(inst, cand_x)
            budget.charge_full(1)

            if use_ls and not budget.exhausted:
                cand_x, df, dv = move_ls(inst, cand_x, budget=budget, rng=rng, max_passes=3)
                f_val += df
                v_val += dv

            if v_val <= 1e-9:
                f_exact = proximity_cost(inst, cand_x)
                if f_exact < best_feasible_f:
                    best_feasible_f = f_exact
                    best_feasible_x = cand_x.copy()
            elif v_val < best_infeasible_v:
                best_infeasible_v = v_val
                fallback_x = cand_x.copy()

            if len(recent_pool) >= 10:
                recent_pool.pop(0)
            recent_pool.append(cand_x)
            div_score = (
                diversity(np.asarray(recent_pool, dtype=np.int32)) if len(recent_pool) >= 2 else 0.0
            )

            tracker.record(
                iteration=iteration,
                best_f_so_far=best_feasible_f,
                diversity=div_score,
                force=(iteration == 0),
            )
            iteration += 1

        tracker.record(
            iteration=max(0, iteration - 1),
            best_f_so_far=best_feasible_f,
            diversity=0.0,
            force=True,
        )

        feasible = best_feasible_x is not None
        return Result(
            best_x=best_feasible_x if feasible else fallback_x,
            best_f=best_feasible_f if feasible else float("inf"),
            feasible=feasible,
            history=tracker.to_array(),
            elapsed=time.perf_counter() - t0,
            evals=budget.used,
            hparams=dict(self.hparams),
        )
