"""Random search baseline solver (`random`) with optional greedy repair.

Repeatedly samples random timetables using an isolated `np.random.Generator`,
optionally repairs hard conflicts, evaluates them within the FEE budget, and
records convergence via `Tracker`.
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


class RandomSearchSolver(Solver):
    """Baseline solver that samples independent random (+ repaired) timetables."""

    name: str = "random"

    def __init__(self, hparams: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize RandomSearchSolver.

        Supported hyperparameters:
            use_repair (bool): Apply greedy conflict repair to each sample (default True).
            max_repair_passes (int): Maximum repair sweeps per sample (default 5).
            diversity_window (int): Recent solutions kept to estimate diversity (default 10).
        """
        defaults = {
            "use_repair": True,
            "max_repair_passes": 5,
            "diversity_window": 10,
        }
        if hparams:
            defaults.update(hparams)
        defaults.update(kwargs)
        super().__init__(defaults)

    def solve(self, inst: Instance, budget: Budget, seed: int) -> Result:
        """Run random search until `budget.exhausted`.

        Args:
            inst: Problem instance in CSR format.
            budget: Fair evaluation budget tracker.
            seed: Deterministic seed for `np.random.default_rng(seed)`.

        Returns:
            Result with best feasible solution (or `best_f = inf` if none found).
        """
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        tracker = Tracker(budget)

        use_repair = bool(self.hparams.get("use_repair", True))
        max_passes = int(self.hparams.get("max_repair_passes", 5))
        div_win = max(2, int(self.hparams.get("diversity_window", 10)))

        best_feasible_x: np.ndarray | None = None
        best_feasible_f: float = float("inf")

        fallback_x: np.ndarray = np.zeros(inst.n_exams, dtype=np.int32)
        best_infeasible_v: float = float("inf")
        best_infeasible_f: float = float("inf")

        recent_pool: list[np.ndarray] = []
        iteration = 0

        while not budget.exhausted:
            if iteration == 0 and use_repair and inst.n_exams >= 200:
                from src.baselines.dsatur import dsatur_construct

                cand_x = dsatur_construct(
                    inst, rng, slot_selection="min_proximity", use_repair=True
                )
            else:
                raw_x = rng.integers(0, inst.n_slots, size=inst.n_exams, dtype=np.int32)
                if use_repair:
                    cand_x = repair(inst, raw_x, rng, max_passes=max_passes)
                else:
                    cand_x = raw_x

            # Charge 1 full evaluation equivalent (FEE) for evaluating cand_x
            v_val = violation_weight(inst, cand_x)
            f_val = proximity_cost(inst, cand_x)
            budget.charge_full(1)

            if v_val == 0.0:
                if f_val < best_feasible_f:
                    best_feasible_f = f_val
                    best_feasible_x = cand_x.copy()
            else:
                if v_val < best_infeasible_v or (
                    v_val == best_infeasible_v and f_val < best_infeasible_f
                ):
                    best_infeasible_v = v_val
                    best_infeasible_f = f_val
                    fallback_x = cand_x.copy()

            if len(recent_pool) >= div_win:
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
        final_x = best_feasible_x if feasible else fallback_x
        final_f = best_feasible_f if feasible else float("inf")

        return Result(
            best_x=final_x,
            best_f=final_f,
            feasible=feasible,
            history=tracker.to_array(),
            elapsed=time.perf_counter() - t0,
            evals=budget.used,
            hparams=dict(self.hparams),
        )
