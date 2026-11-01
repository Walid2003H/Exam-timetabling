"""Simulated Annealing solver (`sa`) with hybrid Move + Kempe-chain neighborhoods.

Follows Kirkpatrick et al. (1983) and Thompson & Dowsland (1998): starts from a
multi-start DSATUR + repaired schedule and explores single-exam relocations and
boundary-evaluated Kempe-chain swaps under a geometric cooling schedule with
reheating on stagnation. Once a feasible timetable is reached, candidate moves
that introduce hard conflicts (`delta_V > 0`) are rejected so search stays
within the feasible coloring manifold.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from src.baselines.dsatur import dsatur_construct
from src.core.instance import Instance
from src.core.problem import (
    delta_move,
    proximity_cost,
    violation_weight,
)
from src.core.solver import Budget, Result, Solver
from src.core.tracker import Tracker
from src.local_search.kempe_ls import apply_kempe_chain_step

_EPS = 1e-9


class SimulatedAnnealingSolver(Solver):
    """Simulated Annealing trajectory solver (`sa`)."""

    name: str = "sa"

    def __init__(self, hparams: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize SimulatedAnnealingSolver.

        Supported hyperparameters:
            initial_temp (float): Starting temperature T_0 (default 1.5).
            final_temp (float): Minimum temperature floor T_min (default 0.002).
            alpha (float): Geometric cooling factor per epoch (default 0.96).
            epoch_length (int): Moves attempted per temperature step (default 80).
            kempe_prob (float): Probability of proposing a Kempe-chain move vs
                a single-exam move (default 0.25).
            warm_starts (int): Initial DSATUR constructions before cooling (default 3).
            reheat_factor (float): Multiplier applied to T when stagnating (default 1.8).
            stagnation_epochs (int): Epochs without improvement before reheating (default 35).
            penalty_weight (float): Penalty lambda if starting state is infeasible (default 1000.0).
        """
        defaults = {
            "initial_temp": 1.5,
            "final_temp": 0.002,
            "alpha": 0.96,
            "epoch_length": 80,
            "kempe_prob": 0.25,
            "warm_starts": 3,
            "reheat_factor": 1.8,
            "stagnation_epochs": 35,
            "penalty_weight": 1000.0,
        }
        if hparams:
            defaults.update(hparams)
        defaults.update(kwargs)
        super().__init__(defaults)

    def solve(self, inst: Instance, budget: Budget, seed: int) -> Result:
        """Run Simulated Annealing within `budget`.

        Args:
            inst: Problem instance in CSR format.
            budget: Fair evaluation budget tracker.
            seed: Deterministic seed for `np.random.default_rng(seed)`.

        Returns:
            Result with the best feasible schedule discovered.
        """
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        tracker = Tracker(budget)

        t_init = float(self.hparams.get("initial_temp", 1.5))
        t_min = float(self.hparams.get("final_temp", 0.002))
        alpha = float(self.hparams.get("alpha", 0.96))
        epoch_len = max(10, int(self.hparams.get("epoch_length", 80)))
        kempe_prob = float(self.hparams.get("kempe_prob", 0.25))
        warm_starts = max(1, int(self.hparams.get("warm_starts", 3)))
        reheat_factor = float(self.hparams.get("reheat_factor", 1.8))
        stag_limit = max(5, int(self.hparams.get("stagnation_epochs", 35)))
        lam = float(self.hparams.get("penalty_weight", 1000.0))

        best_feasible_x: np.ndarray | None = None
        best_feasible_f: float = float("inf")
        current_x: np.ndarray = np.zeros(inst.n_exams, dtype=np.int32)
        cur_v: float = float("inf")
        cur_f: float = float("inf")

        # Warm-start from the best of `warm_starts` DSATUR + repair constructions
        for _ in range(warm_starts):
            if budget.exhausted:
                break
            cand_x = dsatur_construct(inst, rng, slot_selection="min_proximity", use_repair=True)
            v_val = violation_weight(inst, cand_x)
            f_val = proximity_cost(inst, cand_x)
            budget.charge_full(1)

            if v_val <= _EPS and f_val < best_feasible_f:
                best_feasible_f = f_val
                best_feasible_x = cand_x.copy()
            if v_val < cur_v or (abs(v_val - cur_v) <= _EPS and f_val < cur_f):
                cur_v = v_val
                cur_f = f_val
                current_x = cand_x.copy()

        if best_feasible_x is not None:
            current_x = best_feasible_x.copy()
            cur_f = best_feasible_f
            cur_v = 0.0

        fallback_x: np.ndarray = current_x.copy()
        best_infeasible_v: float = cur_v

        tracker.record(iteration=0, best_f_so_far=best_feasible_f, diversity=0.0, force=True)

        temp = t_init
        iteration = 1
        epochs_without_improvement = 0

        while not budget.exhausted and inst.n_slots > 1:
            improved_in_epoch = False

            for _ in range(epoch_len):
                if budget.exhausted:
                    break

                exam_i = int(rng.integers(0, inst.n_exams))
                cur_slot = int(current_x[exam_i])
                cand_slot = int(rng.integers(0, inst.n_slots - 1))
                if cand_slot >= cur_slot:
                    cand_slot += 1

                if rng.random() >= kempe_prob:
                    # O(deg(i)) single-exam move proposal
                    df, dv = delta_move(inst, current_x, exam_i, cand_slot)
                    budget.charge_delta(inst, exam_i)

                    # Once feasible (cur_v == 0), reject moves that create hard conflicts
                    if cur_v <= _EPS and dv > _EPS:
                        accept = False
                    elif cur_v <= _EPS:
                        accept = (df <= 0.0) or (
                            temp > 1e-12 and rng.random() < math.exp(-df / temp)
                        )
                    else:
                        delta_energy = df + lam * (dv / float(inst.n_students))
                        accept = (delta_energy <= 0.0) or (
                            temp > 1e-12 and rng.random() < math.exp(-delta_energy / temp)
                        )

                    if accept:
                        current_x[exam_i] = cand_slot
                        cur_f += df
                        cur_v = max(0.0, cur_v + dv)
                else:
                    # Boundary-evaluated Kempe-chain move proposal (preserves feasibility)
                    applied, df, dv = apply_kempe_chain_step(
                        inst,
                        current_x,
                        exam_i,
                        cand_slot,
                        budget=budget,
                        accept_non_improving=False,
                    )
                    if applied:
                        cur_f += df
                        cur_v = max(0.0, cur_v + dv)
                    elif abs(dv) <= _EPS and df > 0.0 and temp > 1e-12:
                        if rng.random() < math.exp(-df / temp):
                            apply_kempe_chain_step(
                                inst,
                                current_x,
                                exam_i,
                                cand_slot,
                                budget=None,
                                accept_non_improving=True,
                            )
                            cur_f += df

                if cur_v <= _EPS and cur_f < best_feasible_f - _EPS:
                    best_feasible_f = cur_f
                    best_feasible_x = current_x.copy()
                    improved_in_epoch = True
                elif cur_v < best_infeasible_v:
                    best_infeasible_v = cur_v
                    fallback_x = current_x.copy()

                tracker.record(
                    iteration=iteration,
                    best_f_so_far=best_feasible_f,
                    diversity=0.0,
                )
                iteration += 1

            if improved_in_epoch:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            temp = max(t_min, temp * alpha)
            if epochs_without_improvement >= stag_limit:
                temp = min(t_init, max(temp * reheat_factor, t_init * 0.35))
                epochs_without_improvement = 0

        # Final exact verification on best_feasible_x to eliminate any floating-point drift
        if best_feasible_x is not None:
            if violation_weight(inst, best_feasible_x) == 0.0:
                best_feasible_f = proximity_cost(inst, best_feasible_x)
            else:
                best_feasible_x = None
                best_feasible_f = float("inf")

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
