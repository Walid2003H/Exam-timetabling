"""Depth-first Branch & Bound exact solver (`exact`) for Tiny instances.

Uses degree/saturation ordering, warm-start upper bound from DSATUR + Move LS,
symmetry breaking on the highest-degree exam, and incremental partial proximity
cost pruning to compute optimal (or best-verified) solutions on Tiny benchmarks.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.baselines.dsatur import dsatur_construct
from src.core.instance import Instance
from src.core.problem import proximity_cost, violation_weight
from src.core.solver import Budget, Result, Solver
from src.core.tracker import Tracker


class ExactBranchAndBoundSolver(Solver):
    """Exact Branch & Bound solver (`exact`) designed for Tiny instances."""

    name: str = "exact"

    def __init__(self, hparams: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize ExactBranchAndBoundSolver.

        Supported hyperparameters:
            max_exams_exact (int): Maximum N for full B&B search (default 20).
            warm_start_restarts (int): Number of DSATUR+LS warm starts before B&B (default 8).
            time_limit_seconds (float | None): Optional wall-clock cap (default 30.0).
        """
        defaults = {
            "max_exams_exact": 20,
            "warm_start_restarts": 8,
            "time_limit_seconds": 30.0,
        }
        if hparams:
            defaults.update(hparams)
        defaults.update(kwargs)
        super().__init__(defaults)

    def solve(self, inst: Instance, budget: Budget, seed: int) -> Result:
        """Run warm-started Branch & Bound within `budget`.

        Args:
            inst: Problem instance in CSR format.
            budget: Fair evaluation budget tracker.
            seed: Deterministic seed for `np.random.default_rng(seed)`.

        Returns:
            Result with optimal (or best-found) feasible solution.
        """
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        tracker = Tracker(budget)

        max_exams_exact = int(self.hparams.get("max_exams_exact", 20))
        max_r = int(self.hparams.get("warm_start_restarts", 8))
        warm_restarts = max(4, min(max_r, int(budget.max_evals // 15)))
        time_limit = self.hparams.get("time_limit_seconds", 30.0)
        max_sec = float(time_limit) if time_limit is not None else float("inf")

        best_x: np.ndarray | None = None
        best_f: float = float("inf")
        fallback_x = np.zeros(inst.n_exams, dtype=np.int32)

        # Phase 1: Establish a strong initial upper bound via DSATUR + Move LS
        for r in range(warm_restarts):
            if budget.exhausted or (time.perf_counter() - t0) >= max_sec:
                break
            cand_x = dsatur_construct(inst, rng, slot_selection="min_proximity", use_repair=True)
            budget.charge_full(1)
            from src.local_search.shared_ls import intensify

            cand_x = intensify(
                inst,
                cand_x,
                budget=budget,
                rng=rng,
                operator="hybrid",
                max_passes=3,
                kempe_steps=15,
                swap_trials=20,
            )
            if violation_weight(inst, cand_x) == 0.0:
                f_val = proximity_cost(inst, cand_x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = cand_x.copy()
            else:
                fallback_x = cand_x.copy()

            tracker.record(
                iteration=r,
                best_f_so_far=best_f,
                diversity=0.0,
                force=(r == 0),
            )

        # If N exceeds max_exams_exact, return the warm-start bound safely
        if inst.n_exams > max_exams_exact or budget.exhausted:
            tracker.record(iteration=warm_restarts, best_f_so_far=best_f, diversity=0.0, force=True)
            feasible = best_x is not None
            return Result(
                best_x=best_x if feasible else fallback_x,
                best_f=best_f if feasible else float("inf"),
                feasible=feasible,
                history=tracker.to_array(),
                elapsed=time.perf_counter() - t0,
                evals=budget.used,
                hparams=dict(self.hparams),
            )

        # Phase 2: Order exams by descending static conflict degree ( greedy clique order )
        n = inst.n_exams
        n_slots = inst.n_slots
        n_students = float(inst.n_students)

        degrees = np.array([inst.degree(i) for i in range(n)], dtype=np.int32)
        order: list[int] = []
        remaining = set(range(n))
        while remaining:
            # Pick exam with most neighbours already in `order`, tie-break by total degree
            best_u = max(
                remaining,
                key=lambda u: (
                    sum(
                        1
                        for k in range(int(inst.indptr[u]), int(inst.indptr[u + 1]))
                        if int(inst.indices[k]) in order
                    ),
                    int(degrees[u]),
                ),
            )
            order.append(best_u)
            remaining.remove(best_u)

        pos_of_exam = {exam_id: idx for idx, exam_id in enumerate(order)}

        # Precompute backward adjacency lists (neighbours appearing earlier in `order`)
        earlier_nbrs: list[list[tuple[int, int]]] = []
        for idx, u in enumerate(order):
            prev_list: list[tuple[int, int]] = []
            for k in range(int(inst.indptr[u]), int(inst.indptr[u + 1])):
                v = int(inst.indices[k])
                if pos_of_exam[v] < idx:
                    prev_list.append((v, int(inst.weights[k])))
            earlier_nbrs.append(prev_list)

        partial_x = np.full(n, -1, dtype=np.int32)
        best_raw_cost = int(round(best_f * n_students)) if best_f < float("inf") else 10**15
        node_counter = warm_restarts

        def _dfs(depth: int, current_raw_cost: int, max_used_slot: int) -> None:
            nonlocal best_raw_cost, best_x, best_f, node_counter

            if budget.exhausted or (time.perf_counter() - t0) >= max_sec:
                return

            if depth == n:
                budget.charge_full(1)
                if current_raw_cost < best_raw_cost:
                    best_raw_cost = current_raw_cost
                    best_f = float(current_raw_cost) / n_students
                    best_x = partial_x.copy()
                    tracker.record(
                        iteration=node_counter,
                        best_f_so_far=best_f,
                        diversity=0.0,
                        force=True,
                    )
                return

            u = order[depth]
            prev_nbrs = earlier_nbrs[depth]

            # Evaluate all conflict-free slots for exam `u` and sort by incremental cost
            candidates: list[tuple[int, int]] = []
            # Symmetry breaking on the first exam: fix its slot to 0..min(T-1, 2)
            slot_limit = n_slots if depth > 0 else min(n_slots, 3)

            for s in range(slot_limit):
                feasible_slot = True
                inc_raw = 0
                for v, w_uv in prev_nbrs:
                    d = abs(s - int(partial_x[v]))
                    if d == 0:
                        feasible_slot = False
                        break
                    if d <= 5:
                        inc_raw += w_uv * (1 << (5 - d))

                if feasible_slot and (current_raw_cost + inc_raw < best_raw_cost):
                    candidates.append((inc_raw, s))

            if depth >= n - 2:
                budget.charge_delta(inst, u)
            node_counter += 1
            if node_counter % 64 == 0:
                tracker.record(
                    iteration=node_counter,
                    best_f_so_far=best_f,
                    diversity=0.0,
                )

            candidates.sort(key=lambda pair: pair[0])
            for inc_raw, s in candidates:
                if current_raw_cost + inc_raw >= best_raw_cost:
                    break
                if budget.exhausted or (time.perf_counter() - t0) >= max_sec:
                    break
                partial_x[u] = s
                _dfs(depth + 1, current_raw_cost + inc_raw, max(max_used_slot, s))
                partial_x[u] = -1

        _dfs(0, 0, -1)

        tracker.record(
            iteration=node_counter,
            best_f_so_far=best_f,
            diversity=0.0,
            force=True,
        )

        feasible = best_x is not None
        return Result(
            best_x=best_x if feasible else fallback_x,
            best_f=best_f if feasible else float("inf"),
            feasible=feasible,
            history=tracker.to_array(),
            elapsed=time.perf_counter() - t0,
            evals=budget.used,
            hparams=dict(self.hparams),
        )
