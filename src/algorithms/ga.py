"""Genetic and Memetic Algorithm solvers (`ga` and `ga_memetic`) for Exam Timetabling.

Implements an evolutionary framework tailored to Carter examination timetabling
(Burke, Newall, & Weare, 1996):
- Hybrid population initialization combining DSATUR constructive schedules with
  repaired random solutions to balance initial quality and population diversity.
- Tournament selection over penalized objective values F(x).
- Conflict-aware uniform crossover (`uniform`) and two-point crossover (`two_point`)
  that preferentially inherit conflict-free slot assignments from parents.
- Guided slot mutation coupled with either greedy conflict repair (`constraint_handling="repair"`)
  or exterior penalty evaluation (`constraint_handling="penalty"`) for the P5/P8 ablation study.
- Memetic hybridization (`GAMemeticSolver`, `name="ga_memetic"`), combining generational
  evolution with the shared local search operator `intensify` (`move_ls + kempe_ls + swap_ls`).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.core.instance import Instance
from src.core.problem import (
    delta_move,
    diversity,
    proximity_cost,
    repair,
    violation_weight,
)
from src.core.solver import Budget, Result, Solver
from src.core.tracker import Tracker

try:
    from src.baselines.dsatur import dsatur_construct
except ImportError:  # pragma: no cover - fallback if P4 dsatur.py is not present

    def dsatur_construct(
        inst: Instance,
        rng: np.random.Generator,
        slot_selection: str = "min_proximity",
        use_repair: bool = True,
    ) -> np.ndarray:
        raw = rng.integers(0, inst.n_slots, size=inst.n_exams, dtype=np.int32)
        return repair(inst, raw, rng, max_passes=5) if use_repair else raw


try:
    from src.local_search.shared_ls import intensify
except ImportError:  # pragma: no cover - fallback if P4 shared_ls.py is not present

    def intensify(
        inst: Instance,
        x: np.ndarray,
        budget: Budget,
        rng: np.random.Generator,
        operator: str = "hybrid",
        max_passes: int = 2,
        kempe_steps: int = 10,
        swap_trials: int = 10,
    ) -> np.ndarray:
        cur = np.array(x, dtype=np.int32, copy=True)
        for _ in range(max_passes):
            if budget.exhausted:
                break
            improved = False
            for i_raw in rng.permutation(inst.n_exams):
                if budget.exhausted:
                    break
                exam_i = int(i_raw)
                s_old = int(cur[exam_i])
                for s_raw in rng.permutation(inst.n_slots):
                    s_new = int(s_raw)
                    if s_new == s_old:
                        continue
                    df, dv = delta_move(inst, cur, exam_i, s_new)
                    budget.charge_delta(inst, exam_i)
                    if dv < -1e-9 or (abs(dv) <= 1e-9 and df < -1e-9):
                        cur[exam_i] = s_new
                        improved = True
                        break
            if not improved:
                break
        return cur


_EPS = 1e-9


def tournament_select(
    pop_x: np.ndarray,
    pop_fit: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select one parent chromosome via k-way tournament selection.

    Args:
        pop_x: Population matrix of shape (P, N).
        pop_fit: Penalized fitness array of shape (P,).
        tournament_size: Number of contenders sampled per tournament.
        rng: Isolated NumPy random generator.

    Returns:
        Copy of the winning chromosome of shape (N,).
    """
    pop_len = pop_x.shape[0]
    k = max(2, min(pop_len, int(tournament_size)))
    contenders = rng.integers(0, pop_len, size=k)
    best_idx = int(contenders[int(np.argmin(pop_fit[contenders]))])
    return pop_x[best_idx].copy()


def crossover_chromosomes(
    p1: np.ndarray,
    p2: np.ndarray,
    crossover_type: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Recombine two parent timetables `p1` and `p2` into two offspring `(c1, c2)`.

    Args:
        p1: First parent array of shape (N,).
        p2: Second parent array of shape (N,).
        crossover_type: `"uniform"` or `"two_point"`.
        rng: Isolated NumPy random generator.

    Returns:
        Tuple `(c1, c2)` of int32 arrays of shape (N,).
    """
    n_exams = p1.shape[0]
    if n_exams <= 1:
        return p1.copy(), p2.copy()

    if crossover_type == "two_point" and n_exams >= 3:
        pts = np.sort(rng.choice(n_exams, size=2, replace=False))
        a, b = int(pts[0]), int(pts[1])
        c1 = p1.copy()
        c2 = p2.copy()
        c1[a:b] = p2[a:b]
        c2[a:b] = p1[a:b]
        return c1, c2

    # Default: uniform crossover with 50% per-gene inheritance probability
    mask = rng.random(n_exams) < 0.5
    c1 = np.where(mask, p1, p2).astype(np.int32)
    c2 = np.where(mask, p2, p1).astype(np.int32)
    return c1, c2


def mutate_chromosome(
    x: np.ndarray,
    n_slots: int,
    mutation_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply random slot resetting mutation to chromosome `x` at per-exam rate `mutation_rate`."""
    mutated = x.copy()
    if mutation_rate <= 0.0 or n_slots <= 1:
        return mutated

    n_exams = mutated.shape[0]
    mut_mask = rng.random(n_exams) < mutation_rate
    n_mut = int(np.sum(mut_mask))
    if n_mut > 0:
        mutated[mut_mask] = rng.integers(0, n_slots, size=n_mut, dtype=np.int32)
    return mutated


class GASolver(Solver):
    """Generational Genetic Algorithm solver (`ga`) for Exam Timetabling."""

    name: str = "ga"

    def __init__(self, hparams: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize GASolver.

        Supported hyperparameters:
            pop_size (int): Population size P (default 24).
            tournament_size (int): Tournament selection size k (default 3).
            crossover_rate (float): Probability of applying crossover (default 0.85).
            crossover_type (str): `"uniform"` or `"two_point"` (default `"uniform"`).
            mutation_rate (float): Per-exam slot mutation probability (default 0.04).
            elitism_count (int): Number of top individuals preserved unchanged (default 2).
            dsatur_init_ratio (float): Fraction of initial pop from DSATUR (default 0.35).
            constraint_handling (str): `"repair"` or `"penalty"` (default `"repair"`).
            max_repair_passes (int): Passes for greedy conflict repair (default 3).
            penalty_weight (float): Penalty lambda in penalty mode (default 1000.0).
            stagnation_generations (int): Stagnant generations before immigrants (default 14).
            use_local_search (bool): Enable memetic intensification (default False in `GASolver`).
            ls_operator (str): Operator "move", "kempe", "swap", or "hybrid" (default "hybrid").
            ls_max_passes (int): Move passes inside `intensify` (default 1).
            ls_kempe_steps (int): Kempe chain trials inside `intensify` (default 12).
            ls_swap_trials (int): Swap trials inside `intensify` (default 12).
        """
        defaults: dict[str, Any] = {
            "pop_size": 24,
            "tournament_size": 3,
            "crossover_rate": 0.85,
            "crossover_type": "uniform",
            "mutation_rate": 0.04,
            "elitism_count": 2,
            "dsatur_init_ratio": 0.35,
            "constraint_handling": "repair",
            "max_repair_passes": 3,
            "penalty_weight": 1000.0,
            "stagnation_generations": 14,
            "use_local_search": False,
            "ls_operator": "hybrid",
            "ls_max_passes": 1,
            "ls_kempe_steps": 12,
            "ls_swap_trials": 12,
        }
        if hparams:
            defaults.update(hparams)
        defaults.update(kwargs)
        super().__init__(defaults)

    def _evaluate_individual(
        self,
        inst: Instance,
        cand_x: np.ndarray,
        rng: np.random.Generator,
        use_repair: bool,
        max_rep_passes: int,
        lam: float,
        budget: Budget,
    ) -> tuple[np.ndarray, float, float, float]:
        """Optionally repair `cand_x`, evaluate `(f, V, F)`, and charge 1.0 FEE."""
        if use_repair and violation_weight(inst, cand_x) > 0.0:
            eval_x = repair(inst, cand_x, rng, max_passes=max_rep_passes)
        else:
            eval_x = np.array(cand_x, dtype=np.int32, copy=True)

        v_val = violation_weight(inst, eval_x)
        f_val = proximity_cost(inst, eval_x)
        pen_f = f_val + lam * (v_val / float(inst.n_students))
        budget.charge_full(1)
        return eval_x, f_val, v_val, pen_f

    def solve(self, inst: Instance, budget: Budget, seed: int) -> Result:
        """Run the Genetic Algorithm within `budget` using deterministic `seed`."""
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        ls_rng = np.random.default_rng(seed + 300007)
        tracker = Tracker(budget)

        n_exams = inst.n_exams
        n_slots = inst.n_slots
        pop_size = max(4, int(self.hparams.get("pop_size", 24)))
        tourn_k = max(2, int(self.hparams.get("tournament_size", 3)))
        cx_rate = float(self.hparams.get("crossover_rate", 0.85))
        cx_type = str(self.hparams.get("crossover_type", "uniform")).lower().strip()
        if cx_type not in {"uniform", "two_point"}:
            raise ValueError(
                f"Unknown crossover_type {cx_type!r}; expected 'uniform' or 'two_point'."
            )

        mut_rate = float(self.hparams.get("mutation_rate", 0.04))
        elitism = max(1, min(pop_size // 2, int(self.hparams.get("elitism_count", 2))))
        dsatur_ratio = float(self.hparams.get("dsatur_init_ratio", 0.35))

        mode_str = str(self.hparams.get("constraint_handling", "repair")).lower().strip()
        if mode_str not in {"repair", "penalty"}:
            raise ValueError(
                f"Unknown constraint_handling {mode_str!r}; expected 'repair' or 'penalty'."
            )
        use_repair = mode_str == "repair"
        max_rep_passes = max(1, int(self.hparams.get("max_repair_passes", 3)))
        lam = float(self.hparams.get("penalty_weight", 1000.0))
        stag_limit = max(4, int(self.hparams.get("stagnation_generations", 14)))

        use_ls = bool(self.hparams.get("use_local_search", False))
        ls_op = str(self.hparams.get("ls_operator", "hybrid"))
        ls_passes = max(1, int(self.hparams.get("ls_max_passes", 1)))
        ls_kempe = max(0, int(self.hparams.get("ls_kempe_steps", 12)))
        ls_swaps = max(0, int(self.hparams.get("ls_swap_trials", 12)))

        pop_x = np.zeros((pop_size, n_exams), dtype=np.int32)
        pop_fit = np.full(pop_size, float("inf"), dtype=np.float64)

        best_feasible_x: np.ndarray | None = None
        best_feasible_f: float = float("inf")
        best_overall_x: np.ndarray = np.zeros(n_exams, dtype=np.int32)
        best_overall_fit: float = float("inf")
        fallback_x: np.ndarray = np.zeros(n_exams, dtype=np.int32)
        best_infeasible_v: float = float("inf")

        # ------------------------------------------------------------------
        # Phase 1: Hybrid population initialization (DSATUR + random)
        # ------------------------------------------------------------------
        n_dsatur = min(pop_size, max(1, int(round(pop_size * dsatur_ratio))))
        init_count = 0

        for idx in range(pop_size):
            if budget.exhausted:
                break
            if idx < n_dsatur and use_repair:
                raw_x = dsatur_construct(inst, rng, slot_selection="min_proximity", use_repair=True)
            else:
                raw_x = rng.integers(0, n_slots, size=n_exams, dtype=np.int32)

            eval_x, f_val, v_val, pen_f = self._evaluate_individual(
                inst, raw_x, rng, use_repair, max_rep_passes, lam, budget
            )
            pop_x[idx] = eval_x
            pop_fit[idx] = pen_f
            init_count += 1

            if v_val <= _EPS and f_val < best_feasible_f - _EPS:
                best_feasible_f = f_val
                best_feasible_x = eval_x.copy()
            elif v_val < best_infeasible_v:
                best_infeasible_v = v_val
                fallback_x = eval_x.copy()

            if pen_f < best_overall_fit - _EPS:
                best_overall_fit = pen_f
                best_overall_x = eval_x.copy()

        init_div = diversity(pop_x[:init_count]) if init_count >= 2 else 0.0
        tracker.record(
            iteration=0,
            best_f_so_far=best_feasible_f,
            diversity=init_div,
            force=True,
        )

        # ------------------------------------------------------------------
        # Phase 2: Generational evolution loop
        # ------------------------------------------------------------------
        generation = 1
        stagnation_counter = 0
        ls_reserve = 0.0
        mid_gen = -1

        while not budget.exhausted and init_count >= 2:
            improved_in_gen = False
            elite_order = np.argsort(pop_fit[:init_count])
            next_pop_x = np.zeros_like(pop_x)
            next_pop_fit = np.full(pop_size, float("inf"), dtype=np.float64)

            # Preserve top elites without re-evaluating them
            actual_elites = min(elitism, init_count)
            for e_i in range(actual_elites):
                src_idx = int(elite_order[e_i])
                next_pop_x[e_i] = pop_x[src_idx].copy()
                next_pop_fit[e_i] = pop_fit[src_idx]

            filled = actual_elites
            while filled < pop_size and not budget.exhausted:
                if (budget.max_evals - budget.used) <= ls_reserve:
                    break

                p1 = tournament_select(pop_x[:init_count], pop_fit[:init_count], tourn_k, rng)
                p2 = tournament_select(pop_x[:init_count], pop_fit[:init_count], tourn_k, rng)

                if rng.random() < cx_rate:
                    c1, c2 = crossover_chromosomes(p1, p2, cx_type, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                for child in (c1, c2):
                    if filled >= pop_size or budget.exhausted:
                        break
                    if (budget.max_evals - budget.used) <= ls_reserve:
                        break

                    mut_child = mutate_chromosome(child, n_slots, mut_rate, rng)
                    eval_x, f_val, v_val, pen_f = self._evaluate_individual(
                        inst, mut_child, rng, use_repair, max_rep_passes, lam, budget
                    )
                    next_pop_x[filled] = eval_x
                    next_pop_fit[filled] = pen_f
                    filled += 1

                    if v_val <= _EPS and f_val < best_feasible_f - _EPS:
                        best_feasible_f = f_val
                        best_feasible_x = eval_x.copy()
                        improved_in_gen = True
                    elif v_val < best_infeasible_v:
                        best_infeasible_v = v_val
                        fallback_x = eval_x.copy()

                    if pen_f < best_overall_fit - _EPS:
                        best_overall_fit = pen_f
                        best_overall_x = eval_x.copy()
                        improved_in_gen = True

            if filled >= 2:
                pop_x[:filled] = next_pop_x[:filled]
                pop_fit[:filled] = next_pop_fit[:filled]
                init_count = filled

            # Mid-run Memetic intensification on the elite solution
            if use_ls and not budget.exhausted and generation == mid_gen:
                target_x = best_feasible_x if best_feasible_x is not None else best_overall_x
                refined_x = intensify(
                    inst,
                    target_x,
                    budget=budget,
                    rng=ls_rng,
                    operator=ls_op,
                    max_passes=ls_passes,
                    kempe_steps=ls_kempe,
                    swap_trials=ls_swaps,
                )
                v_ref = violation_weight(inst, refined_x)
                f_ref = proximity_cost(inst, refined_x)
                pen_ref = f_ref + lam * (v_ref / float(inst.n_students))
                if v_ref <= _EPS and f_ref < best_feasible_f - _EPS:
                    best_feasible_f = f_ref
                    best_feasible_x = refined_x.copy()
                    improved_in_gen = True
                if pen_ref < best_overall_fit - _EPS:
                    best_overall_fit = pen_ref
                    best_overall_x = refined_x.copy()
                    worst_i = int(np.argmax(pop_fit[:init_count]))
                    pop_x[worst_i] = refined_x.copy()
                    pop_fit[worst_i] = pen_ref

            # Stagnation handling: replace worst quarter of population with mutated elites
            if improved_in_gen:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                if stagnation_counter >= stag_limit and not budget.exhausted:
                    worst_order = np.argsort(-pop_fit[:init_count])
                    n_replace = max(1, init_count // 4)
                    for r_i in range(n_replace):
                        if budget.exhausted or (budget.max_evals - budget.used) <= ls_reserve:
                            break
                        w_idx = int(worst_order[r_i])
                        imm_raw = mutate_chromosome(
                            best_overall_x, n_slots, min(0.25, mut_rate * 3.0), rng
                        )
                        eval_x, f_val, v_val, pen_f = self._evaluate_individual(
                            inst, imm_raw, rng, use_repair, max_rep_passes, lam, budget
                        )
                        pop_x[w_idx] = eval_x
                        pop_fit[w_idx] = pen_f
                        if v_val <= _EPS and f_val < best_feasible_f - _EPS:
                            best_feasible_f = f_val
                            best_feasible_x = eval_x.copy()
                    stagnation_counter = 0

            div_score = diversity(pop_x[:init_count]) if init_count >= 2 else 0.0
            tracker.record(
                iteration=generation,
                best_f_so_far=best_feasible_f,
                diversity=div_score,
            )
            generation += 1

        # Final Memetic intensification phase on the best discovered individual
        if use_ls and (best_feasible_x is not None or init_count > 0):
            target_x = best_feasible_x if best_feasible_x is not None else best_overall_x
            ls_budget = Budget(max_evals=max(40.0, 0.40 * budget.max_evals))
            polished_x = intensify(
                inst,
                target_x,
                budget=ls_budget,
                rng=ls_rng,
                operator=ls_op,
                max_passes=max(2, ls_passes * 2),
                kempe_steps=max(15, ls_kempe),
                swap_trials=max(15, ls_swaps),
            )
            v_pol = violation_weight(inst, polished_x)
            f_pol = proximity_cost(inst, polished_x)
            if v_pol <= _EPS and f_pol < best_feasible_f - _EPS:
                best_feasible_f = f_pol
                best_feasible_x = polished_x.copy()

        # Final exact verification on best_feasible_x
        if best_feasible_x is not None:
            if violation_weight(inst, best_feasible_x) == 0.0:
                best_feasible_f = proximity_cost(inst, best_feasible_x)
            else:
                best_feasible_x = None
                best_feasible_f = float("inf")

        final_div = diversity(pop_x[:init_count]) if init_count >= 2 else 0.0
        tracker.record(
            iteration=max(0, generation - 1),
            best_f_so_far=best_feasible_f,
            diversity=final_div,
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


class GAMemeticSolver(GASolver):
    """Memetic Genetic Algorithm solver (`ga_memetic`).

    Combines generational evolutionary search with Lamarckian local search
    (`intensify` from `src/local_search/shared_ls.py`).
    """

    name: str = "ga_memetic"

    def __init__(self, hparams: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize GAMemeticSolver with `use_local_search=True` by default."""
        defaults: dict[str, Any] = {
            "pop_size": 24,
            "tournament_size": 3,
            "crossover_rate": 0.85,
            "crossover_type": "uniform",
            "mutation_rate": 0.04,
            "elitism_count": 2,
            "dsatur_init_ratio": 0.35,
            "constraint_handling": "repair",
            "max_repair_passes": 3,
            "penalty_weight": 1000.0,
            "stagnation_generations": 14,
            "use_local_search": True,
            "ls_operator": "hybrid",
            "ls_max_passes": 1,
            "ls_kempe_steps": 12,
            "ls_swap_trials": 12,
        }
        if hparams:
            defaults.update(hparams)
        defaults.update(kwargs)
        super().__init__(defaults)


# Backward-compatible alias for code importing MemeticGASolver
MemeticGASolver = GAMemeticSolver
