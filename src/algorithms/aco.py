"""MAX-MIN Ant System solvers (`aco` and `aco_memetic`) for Exam Timetabling.

Implements the MAX-MIN Ant System (MMAS) proposed by Stuetzle & Hoos (2000)
and adapted to university timetabling by Socha, Knowles, & Samples (2002):
- Each ant constructs a complete timetable step by step, ordering exams via a
  randomized saturation/static-degree rule inspired by DSATUR.
- When assigning exam `u` to a time slot `s`, the ant samples from candidate
  slots with probability proportional to `(tau[u, s] ** alpha) * (eta[u, s] ** beta)`,
  where `tau[u, s]` is the pheromone trail on pair `(u, s)` and `eta[u, s]` is the
  local heuristic desirability inversely related to conflict weight and Carter
  proximity cost against already-scheduled neighbours of `u`.
- Pheromone trails are evaporated at rate `rho` and reinforced by either the
  iteration-best ant or the global-best feasible ant, then clamped to the dynamic
  interval `[tau_min, tau_max]` to prevent premature stagnation.
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
except ImportError:  # pragma: no cover - fallback if P4 dsatur.py is not yet merged

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
except ImportError:  # pragma: no cover - fallback if P4 shared_ls.py is not yet merged

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


def compute_mmas_bounds(
    best_cost: float,
    rho: float,
    n_slots: int,
    p_best: float = 0.05,
) -> tuple[float, float]:
    """Compute MMAS upper and lower pheromone bounds `(tau_min, tau_max)`.

    Follows Stuetzle & Hoos (2000):
      - `tau_max = 1 / (rho * max(1.0, best_cost))`
      - `tau_min = tau_max * (1 - p_best**(1/n)) / ((avg - 1) * p_best**(1/n))`
        bounded into `[0.005 * tau_max, 0.25 * tau_max]` for numerical stability.

    Args:
        best_cost: Best objective cost discovered so far.
        rho: Pheromone evaporation rate in (0, 1).
        n_slots: Number of available time slots T.
        p_best: Target probability of constructing the best tour at convergence.

    Returns:
        Tuple `(tau_min, tau_max)` with `0 < tau_min < tau_max`.
    """
    safe_cost = max(1.0, float(best_cost)) if np.isfinite(best_cost) else 50.0
    safe_rho = max(0.01, min(0.99, float(rho)))
    tau_max = 1.0 / (safe_rho * safe_cost)

    avg = max(2.0, 0.5 * float(n_slots))
    root_p = float(p_best) ** (1.0 / max(2.0, float(n_slots)))
    denom = max(1e-6, (avg - 1.0) * root_p)
    raw_ratio = (1.0 - root_p) / denom
    clamped_ratio = max(0.01, min(0.20, raw_ratio))
    tau_min = tau_max * clamped_ratio
    return float(tau_min), float(tau_max)


def construct_ant_timetable(
    inst: Instance,
    tau: np.ndarray,
    alpha: float,
    beta: float,
    q0: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct a complete timetable for a single ant using pheromone + DSATUR heuristic.

    Exams are ordered by dynamic saturation degree + static degree + small uniform
    jitter so difficult exams are scheduled first. For each exam `u`:
      1. Computes conflict weight `conf[s]` and Carter proximity cost `prox[s]` for
         every slot `s in 0..T-1` against already-scheduled neighbours of `u`.
      2. Restricts selection to minimum-conflict slots (zero-conflict slots whenever
         at least one conflict-free slot exists) and defines the heuristic
         desirability `eta[s] = 1.0 / (1.0 + prox[s] / max(1, S) + 100.0 * conf[s])`.
      3. Applies pseudo-random proportional selection (with greedy exploitation
         probability `q0` and roulette-wheel sampling probability `1 - q0` over
         `(tau[u, s] ** alpha) * (eta[s] ** beta)`).

    Note: Per Section 3.3 of the Interface Contract, heuristic calculations inside
    ACO ant construction cost 0.0 FEE; 1.0 FEE is charged when the completed ant
    timetable is evaluated.

    Args:
        inst: Problem instance in CSR format.
        tau: Pheromone matrix of shape (N, T).
        alpha: Pheromone influence exponent.
        beta: Heuristic desirability exponent.
        q0: Pseudo-random greedy exploitation probability in [0, 1].
        rng: Isolated NumPy random generator.

    Returns:
        Constructed timetable array of shape (N,) and dtype int32.
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

    nbr_slot_used = np.zeros((n_exams, n_slots), dtype=bool)
    sat_deg = np.zeros(n_exams, dtype=np.int32)
    norm_s = float(max(1, inst.n_students))

    for _ in range(n_exams):
        # Select next unassigned exam with highest (saturation_degree, static_degree, noise)
        scores = np.where(unassigned, sat_deg * 1000000.0 + degrees * 100.0 + tie_noise, -1.0)
        u = int(np.argmax(scores))

        nbrs = indices[indptr[u] : indptr[u + 1]]
        wts = weights[indptr[u] : indptr[u + 1]]
        assigned_mask = ~unassigned[nbrs]
        used_nbr_slots = x[nbrs[assigned_mask]]
        used_nbr_wts = wts[assigned_mask]

        if used_nbr_slots.shape[0] == 0:
            # No neighbours scheduled yet: sample purely from pheromone trail tau[u]
            probs = np.power(np.maximum(tau[u], 1e-12), alpha)
            total_p = float(np.sum(probs))
            if total_p <= 0.0:
                chosen_slot = int(rng.integers(0, n_slots))
            else:
                probs /= total_p
                chosen_slot = int(rng.choice(n_slots, p=probs))
        else:
            slot_conf = np.bincount(used_nbr_slots, weights=used_nbr_wts, minlength=n_slots)
            min_c = np.min(slot_conf)
            cand_slots = np.flatnonzero(slot_conf == min_c)

            if cand_slots.shape[0] == 1:
                chosen_slot = int(cand_slots[0])
            else:
                # Compute Carter proximity penalty on candidate minimum-conflict slots
                diffs = np.abs(cand_slots[:, None] - used_nbr_slots[None, :])
                prox_mask = (diffs >= 1) & (diffs <= 5)
                prox_pen = np.where(
                    prox_mask,
                    (1 << np.clip(5 - diffs, 0, 5)) * used_nbr_wts[None, :],
                    0,
                )
                prox_sums = np.sum(prox_pen, axis=1).astype(np.float64) / norm_s
                eta_vals = 1.0 / (1.0 + prox_sums)
                tau_vals = np.maximum(tau[u, cand_slots], 1e-12)

                desirability = np.power(tau_vals, alpha) * np.power(eta_vals, beta)
                if rng.random() < q0:
                    chosen_slot = int(cand_slots[int(np.argmax(desirability))])
                else:
                    sum_d = float(np.sum(desirability))
                    if sum_d <= 0.0 or not np.isfinite(sum_d):
                        chosen_slot = int(cand_slots[int(rng.integers(0, cand_slots.shape[0]))])
                    else:
                        probs = desirability / sum_d
                        chosen_slot = int(rng.choice(cand_slots, p=probs))

        x[u] = chosen_slot
        unassigned[u] = False

        # Increment saturation degree for unscheduled neighbours of u
        unassigned_nbrs = nbrs[unassigned[nbrs]]
        new_sat = unassigned_nbrs[~nbr_slot_used[unassigned_nbrs, chosen_slot]]
        nbr_slot_used[new_sat, chosen_slot] = True
        sat_deg[new_sat] += 1

    return x


def update_mmas_pheromones(
    tau: np.ndarray,
    elite_x: np.ndarray,
    elite_cost: float,
    rho: float,
    tau_min: float,
    tau_max: float,
) -> None:
    """Evaporate pheromone matrix tau, deposit on elite_x, and clamp to [tau_min, tau_max]."""
    tau *= 1.0 - float(rho)
    deposit = 1.0 / max(1.0, float(elite_cost))
    exam_indices = np.arange(elite_x.shape[0])
    tau[exam_indices, elite_x] += deposit
    np.clip(tau, tau_min, tau_max, out=tau)


class ACOSolver(Solver):
    """MAX-MIN Ant System solver (`aco`) for the Examination Timetabling Problem."""

    name: str = "aco"

    def __init__(self, hparams: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize ACOSolver.

        Supported hyperparameters:
            n_ants (int): Number of ants per colony iteration (default 16).
            alpha (float): Pheromone trail exponent (default 1.0).
            beta (float): Heuristic desirability exponent (default 2.5).
            rho (float): Pheromone evaporation rate in (0, 1) (default 0.12).
            q0 (float): Greedy exploitation probability in [0, 1] (default 0.15).
            p_best (float): MMAS convergence probability parameter (default 0.05).
            gbest_deposit_freq (int): Deposit global-best every k-th iteration (default 3).
            constraint_handling (str): `"repair"` or `"penalty"` (default `"repair"`).
            max_repair_passes (int): Passes for greedy conflict repair (default 3).
            penalty_weight (float): Penalty lambda in penalty mode (default 1000.0).
            stagnation_iters (int): Stagnant iterations before pheromone smoothing (default 14).
            smooth_factor (float): Trail smoothing factor delta toward tau_max (default 0.35).
            use_local_search (bool): Enable memetic intensification (default False in `ACOSolver`).
            ls_operator (str): Operator "move", "kempe", "swap", or "hybrid" (default "hybrid").
            ls_top_k (int): Number of top ants refined per iteration in memetic mode (default 1).
            ls_max_passes (int): Move passes inside `intensify` (default 1).
            ls_kempe_steps (int): Kempe chain trials inside `intensify` (default 12).
            ls_swap_trials (int): Swap trials inside `intensify` (default 12).
        """
        defaults: dict[str, Any] = {
            "n_ants": 16,
            "alpha": 1.0,
            "beta": 2.5,
            "rho": 0.12,
            "q0": 0.15,
            "p_best": 0.05,
            "gbest_deposit_freq": 3,
            "constraint_handling": "repair",
            "max_repair_passes": 3,
            "penalty_weight": 1000.0,
            "stagnation_iters": 14,
            "smooth_factor": 0.35,
            "use_local_search": False,
            "ls_operator": "hybrid",
            "ls_top_k": 1,
            "ls_max_passes": 1,
            "ls_kempe_steps": 12,
            "ls_swap_trials": 12,
        }
        if hparams:
            defaults.update(hparams)
        defaults.update(kwargs)
        super().__init__(defaults)

    def solve(self, inst: Instance, budget: Budget, seed: int) -> Result:
        """Run MAX-MIN Ant System within `budget` using deterministic `seed`."""
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        ls_rng = np.random.default_rng(seed + 200003)
        tracker = Tracker(budget)

        n_exams = inst.n_exams
        n_slots = inst.n_slots
        n_ants = max(2, int(self.hparams.get("n_ants", 16)))
        alpha = float(self.hparams.get("alpha", 1.0))
        beta = float(self.hparams.get("beta", 2.5))
        rho = float(self.hparams.get("rho", 0.12))
        q0 = float(self.hparams.get("q0", 0.15))
        p_best = float(self.hparams.get("p_best", 0.05))
        gbest_freq = max(1, int(self.hparams.get("gbest_deposit_freq", 3)))

        mode_str = str(self.hparams.get("constraint_handling", "repair")).lower().strip()
        if mode_str not in {"repair", "penalty"}:
            raise ValueError(
                f"Unknown constraint_handling {mode_str!r}; expected 'repair' or 'penalty'."
            )
        use_repair = mode_str == "repair"
        max_rep_passes = max(1, int(self.hparams.get("max_repair_passes", 3)))
        lam = float(self.hparams.get("penalty_weight", 1000.0))
        stag_limit = max(4, int(self.hparams.get("stagnation_iters", 14)))
        smooth_factor = float(self.hparams.get("smooth_factor", 0.35))

        use_ls = bool(self.hparams.get("use_local_search", False))
        ls_op = str(self.hparams.get("ls_operator", "hybrid"))
        ls_passes = max(1, int(self.hparams.get("ls_max_passes", 1)))
        ls_kempe = max(0, int(self.hparams.get("ls_kempe_steps", 12)))
        ls_swaps = max(0, int(self.hparams.get("ls_swap_trials", 12)))

        # ------------------------------------------------------------------
        # Phase 1: Initial DSATUR seed solution to calibrate MMAS tau_max
        # ------------------------------------------------------------------
        seed_x = dsatur_construct(inst, rng, slot_selection="min_proximity", use_repair=use_repair)
        seed_v = violation_weight(inst, seed_x)
        seed_f = proximity_cost(inst, seed_x)
        seed_pen = seed_f + lam * (seed_v / float(inst.n_students))
        budget.charge_full(1)

        best_feasible_x: np.ndarray | None = seed_x.copy() if seed_v <= _EPS else None
        best_feasible_f: float = seed_f if seed_v <= _EPS else float("inf")
        gbest_x: np.ndarray = seed_x.copy()
        gbest_fit: float = seed_pen
        fallback_x: np.ndarray = seed_x.copy()
        best_infeasible_v: float = seed_v

        tau_min, tau_max = compute_mmas_bounds(gbest_fit, rho, n_slots, p_best=p_best)
        # Initialize all pheromone trails to tau_max per MMAS standard (Stuetzle & Hoos, 2000)
        tau = np.full((n_exams, n_slots), tau_max, dtype=np.float64)
        update_mmas_pheromones(tau, gbest_x, gbest_fit, rho, tau_min, tau_max)

        tracker.record(
            iteration=0,
            best_f_so_far=best_feasible_f,
            diversity=0.0,
            force=True,
        )

        # ------------------------------------------------------------------
        # Phase 2: Colony construction, optional memetic LS, and pheromone update
        # ------------------------------------------------------------------
        iteration = 1
        stagnation_counter = 0
        colony_x = np.zeros((n_ants, n_exams), dtype=np.int32)
        colony_fit = np.full(n_ants, float("inf"), dtype=np.float64)

        ls_reserve = 0.06 * budget.max_evals if use_ls else 0.0
        while not budget.exhausted and (budget.max_evals - budget.used) > ls_reserve:
            improved_in_iter = False
            built_ants = 0

            for k in range(n_ants):
                if budget.exhausted:
                    break

                raw_x = construct_ant_timetable(inst, tau, alpha=alpha, beta=beta, q0=q0, rng=rng)
                if use_repair and violation_weight(inst, raw_x) > 0.0:
                    ant_x = repair(inst, raw_x, rng, max_passes=max_rep_passes)
                else:
                    ant_x = raw_x

                v_val = violation_weight(inst, ant_x)
                f_val = proximity_cost(inst, ant_x)
                pen_f = f_val + lam * (v_val / float(inst.n_students))
                budget.charge_full(1)

                colony_x[k] = ant_x
                colony_fit[k] = pen_f
                built_ants += 1

                if v_val <= _EPS and f_val < best_feasible_f - _EPS:
                    best_feasible_f = f_val
                    best_feasible_x = ant_x.copy()
                    improved_in_iter = True
                elif v_val < best_infeasible_v:
                    best_infeasible_v = v_val
                    fallback_x = ant_x.copy()

                if pen_f < gbest_fit - _EPS:
                    gbest_fit = pen_f
                    gbest_x = ant_x.copy()
                    improved_in_iter = True

            if built_ants == 0:
                break

            # Mid-run Memetic Local Search on gbest_x once 50% of budget is reached
            mid_iter = max(2, int(0.45 * budget.max_evals / n_ants))
            if use_ls and not budget.exhausted and iteration == mid_iter:
                refined_x = intensify(
                    inst,
                    gbest_x,
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
                if pen_ref < gbest_fit - _EPS:
                    gbest_fit = pen_ref
                    gbest_x = refined_x.copy()
                    improved_in_iter = True

            # Update MMAS bounds and reinforce pheromones (alternating ibest and gbest)
            ibest_idx = int(np.argmin(colony_fit[:built_ants]))
            ibest_x = colony_x[ibest_idx]
            ibest_fit = float(colony_fit[ibest_idx])

            tau_min, tau_max = compute_mmas_bounds(gbest_fit, rho, n_slots, p_best=p_best)
            if iteration % gbest_freq == 0:
                update_mmas_pheromones(tau, gbest_x, gbest_fit, rho, tau_min, tau_max)
            else:
                update_mmas_pheromones(tau, ibest_x, ibest_fit, rho, tau_min, tau_max)

            # Trail smoothing when colony stagnates (Stuetzle & Hoos, 2000)
            if improved_in_iter:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                if stagnation_counter >= stag_limit:
                    tau += smooth_factor * (tau_max - tau)
                    np.clip(tau, tau_min, tau_max, out=tau)
                    stagnation_counter = 0

            div_score = diversity(colony_x[:built_ants]) if built_ants >= 2 else 0.0
            tracker.record(
                iteration=iteration,
                best_f_so_far=best_feasible_f,
                diversity=div_score,
            )
            iteration += 1

        # Final Memetic intensification phase using the reserved budget
        if use_ls and not budget.exhausted:
            target_x = best_feasible_x if best_feasible_x is not None else gbest_x
            polished_x = intensify(
                inst,
                target_x,
                budget=budget,
                rng=ls_rng,
                operator=ls_op,
                max_passes=max(3, ls_passes * 3),
                kempe_steps=max(25, ls_kempe * 3),
                swap_trials=max(30, ls_swaps * 3),
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

        final_div = diversity(colony_x[:built_ants]) if built_ants >= 2 else 0.0
        tracker.record(
            iteration=max(0, iteration - 1),
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


class ACOMemeticSolver(ACOSolver):
    """Memetic MAX-MIN Ant System solver (`aco_memetic`).

    Couples MMAS pheromone-guided construction with the shared local search
    operator `intensify` (`move_ls + kempe_ls + swap_ls`) applied to the best
    ant(s) constructed in each iteration before pheromone reinforcement.
    """

    name: str = "aco_memetic"

    def __init__(self, hparams: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize ACOMemeticSolver with `use_local_search=True` by default."""
        defaults: dict[str, Any] = {
            "n_ants": 16,
            "alpha": 1.0,
            "beta": 2.5,
            "rho": 0.12,
            "q0": 0.15,
            "p_best": 0.05,
            "gbest_deposit_freq": 3,
            "constraint_handling": "repair",
            "max_repair_passes": 3,
            "penalty_weight": 1000.0,
            "stagnation_iters": 14,
            "smooth_factor": 0.35,
            "use_local_search": True,
            "ls_operator": "hybrid",
            "ls_top_k": 1,
            "ls_max_passes": 1,
            "ls_kempe_steps": 8,
            "ls_swap_trials": 8,
        }
        if hparams:
            defaults.update(hparams)
        defaults.update(kwargs)
        super().__init__(defaults)
