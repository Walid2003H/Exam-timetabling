"""Particle Swarm Optimization solvers (`pso` and `pso_memetic`) for Exam Timetabling.

Implements two discrete/combinatorial PSO encodings required by the project plan:
1. Random-Keys PSO (`encoding="random_keys"`, Bean 1994; Kennedy & Eberhart 1995):
   Each particle maintains a continuous position vector z in [0, 1]^N and velocity v.
   In `"priority_greedy"` decoding mode, keys perturb the DSATUR static degree ordering
   and each exam is assigned to its lowest-conflict, lowest-proximity slot. In `"slot_map"`
   decoding mode, continuous keys are mapped directly to slot indices floor(z * T).
2. Direct Discrete PSO (`encoding="direct_discrete"`):
   Each particle operates directly on slot vectors x in {0, ..., T-1}^N and updates
   positions via velocity-modulated probabilistic inheritance from personal best (`pbest`),
   global best (`gbest`), inertia (`x`), and random exploration.

Supports both `"repair"` and `"penalty"` constraint-handling modes for ablation studies,
and provides `PSOMemeticSolver` (`name="pso_memetic"`), which couples swarm search with
the shared local search operator `intensify` from `src/local_search/shared_ls.py`.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.baselines.dsatur import dsatur_construct
from src.core.instance import Instance
from src.core.problem import (
    diversity,
    get_proximity_kernel,
    proximity_cost,
    repair,
    violation_weight,
)
from src.core.solver import Budget, Result, Solver
from src.core.tracker import Tracker
from src.local_search.shared_ls import intensify

_EPS = 1e-9


def decode_random_keys(
    inst: Instance,
    keys: np.ndarray,
    rng: np.random.Generator,
    decode_mode: str = "priority_greedy",
) -> np.ndarray:
    """Decode a continuous random-key vector `keys` in [0, 1]^N into a discrete timetable.

    Args:
        inst: Problem instance in CSR format.
        keys: Continuous vector of shape (N,) with values in [0.0, 1.0].
        rng: Isolated NumPy random generator for tie-breaking.
        decode_mode: Either `"priority_greedy"` (keys modulate exam scheduling
            priority and slot selection minimizes conflict + Carter proximity cost)
            or `"slot_map"` (direct quantization `floor(keys * T)`).

    Returns:
        Discrete slot assignment array of shape (N,) and dtype int32.

    Time Complexity:
        O(N log N + |E| * T) for `"priority_greedy"`, O(N) for `"slot_map"`.
    """
    n_exams = inst.n_exams
    n_slots = inst.n_slots

    if decode_mode == "slot_map":
        scaled = np.floor(np.clip(keys, 0.0, 0.999999) * n_slots).astype(np.int32)
        return scaled

    # Priority-greedy decoding: combine static conflict degree with continuous key
    indptr = inst.indptr
    indices = inst.indices
    weights = inst.weights
    degrees = np.diff(indptr).astype(np.float64)
    max_deg = float(np.max(degrees)) if n_exams > 0 else 1.0
    norm_deg = degrees / max(1.0, max_deg)

    # Higher key + higher normalized degree -> scheduled earlier
    priority = 0.65 * norm_deg + 0.35 * keys
    order = np.argsort(-priority, kind="mergesort")

    x = np.full(n_exams, -1, dtype=np.int32)
    slot_table = np.zeros((n_exams, n_slots), dtype=np.int32)
    k_prox = get_proximity_kernel(inst)

    for u_raw in order:
        u = int(u_raw)
        row = slot_table[u]
        min_c = int(np.min(row))
        cand_slots = np.flatnonzero(row == min_c)

        if cand_slots.shape[0] == n_slots:
            frac = float(keys[u] * n_slots) - int(keys[u] * n_slots)
            chosen_slot = min(n_slots - 1, max(0, int(frac * n_slots)))
        elif cand_slots.shape[0] == 1:
            chosen_slot = int(cand_slots[0])
        else:
            key_bias = np.abs(cand_slots / float(n_slots) - float(keys[u])) * 1e-2
            scores = (k_prox[cand_slots] @ row) + key_bias + rng.random(cand_slots.shape[0]) * 1e-4
            chosen_slot = int(cand_slots[int(np.argmin(scores))])

        x[u] = chosen_slot
        nbrs = indices[indptr[u] : indptr[u + 1]]
        wts = weights[indptr[u] : indptr[u + 1]]
        slot_table[nbrs, chosen_slot] += wts

    return x


def encode_slots_to_keys(
    inst: Instance,
    x: np.ndarray,
    rng: np.random.Generator,
    decode_mode: str = "priority_greedy",
) -> np.ndarray:
    """Map a discrete timetable `x` back to a continuous random-key vector in [0, 1]^N.

    Used during swarm initialization and Lamarckian memetic write-back so continuous
    particle positions reflect repaired or locally refined schedules.
    """
    n_slots = max(1, inst.n_slots)
    if decode_mode == "slot_map":
        jitter = rng.uniform(0.15, 0.85, size=inst.n_exams)
        return np.clip((x.astype(np.float64) + jitter) / float(n_slots), 0.0, 1.0)

    # Map slots to smooth sub-intervals in [0, 1] with small random jitter
    jitter = rng.uniform(0.1, 0.9, size=inst.n_exams)
    return np.clip((x.astype(np.float64) + jitter) / float(n_slots), 0.0, 1.0)


def update_direct_discrete_particle(
    x: np.ndarray,
    vel_prob: np.ndarray,
    pbest_x: np.ndarray,
    gbest_x: np.ndarray,
    n_slots: int,
    w: float,
    c1: float,
    c2: float,
    mutation_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Update a Direct Discrete PSO particle's velocity probabilities and slot vector.

    Adapts the standard PSO velocity equation to discrete slot assignments:
    - Each exam dimension maintains a velocity propensity `vel_prob[i]` in [0.05, 0.95]
      driven by disagreements `(pbest_x != x)` and `(gbest_x != x)`.
    - For dimensions triggered to move, the new slot is copied from `gbest_x`
      (social component `c2`), `pbest_x` (cognitive component `c1`), or a random
      slot (turbulence `mutation_rate`).

    Args:
        x: Current discrete timetable of shape (N,).
        vel_prob: Current per-exam move probability vector of shape (N,).
        pbest_x: Personal best timetable of shape (N,).
        gbest_x: Global best timetable of shape (N,).
        n_slots: Number of available time slots T.
        w: Inertia weight scaling prior velocity propensity.
        c1: Cognitive acceleration coefficient toward `pbest_x`.
        c2: Social acceleration coefficient toward `gbest_x`.
        mutation_rate: Probability of random slot perturbation per exam.
        rng: Isolated NumPy random generator.

    Returns:
        Tuple `(new_x, new_vel_prob)`.
    """
    n_exams = x.shape[0]
    r1 = rng.random(n_exams)
    r2 = rng.random(n_exams)

    diff_p = (pbest_x != x).astype(np.float64)
    diff_g = (gbest_x != x).astype(np.float64)

    # Continuous propensity update normalized to [0.05, 0.95]
    raw_v = w * vel_prob + c1 * r1 * diff_p + c2 * r2 * diff_g
    norm_scale = max(1.0, w + c1 + c2)
    new_vel = np.clip(raw_v / norm_scale, 0.05, 0.95)

    new_x = x.copy()
    move_mask = rng.random(n_exams) < new_vel

    # Split moving dimensions between cognitive (pbest) and social (gbest) attractors
    social_share = c2 / max(1e-6, c1 + c2)
    choose_gbest = rng.random(n_exams) < social_share

    pbest_mask = move_mask & (~choose_gbest)
    gbest_mask = move_mask & choose_gbest
    new_x[pbest_mask] = pbest_x[pbest_mask]
    new_x[gbest_mask] = gbest_x[gbest_mask]

    # Small random slot turbulence to prevent premature swarm collapse
    if mutation_rate > 0.0:
        mut_mask = rng.random(n_exams) < mutation_rate
        n_mut = int(np.sum(mut_mask))
        if n_mut > 0:
            new_x[mut_mask] = rng.integers(0, n_slots, size=n_mut, dtype=np.int32)

    return new_x, new_vel


class PSOSolver(Solver):
    """Particle Swarm Optimization solver (`pso`) with Random-Keys and Discrete modes."""

    name: str = "pso"

    def __init__(self, hparams: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize PSOSolver.

        Supported hyperparameters:
            swarm_size (int): Number of particles K in the swarm (default 24).
            encoding (str): `"random_keys"` or `"direct_discrete"` (default `"random_keys"`).
            decode_mode (str): "priority_greedy" or "slot_map" (default "priority_greedy").
            w_max (float): Initial inertia weight (default 0.78).
            w_min (float): Final inertia weight (default 0.42).
            c1 (float): Cognitive acceleration coefficient (default 1.49).
            c2 (float): Social acceleration coefficient (default 1.49).
            v_max (float): Maximum continuous velocity magnitude (default 0.35).
            mutation_rate (float): Turbulence probability per exam (default 0.03).
            dsatur_init_ratio (float): Fraction of initial swarm seeded via DSATUR (default 0.35).
            constraint_handling (str): `"repair"` or `"penalty"` (default `"repair"`).
            max_repair_passes (int): Passes for greedy conflict repair (default 4).
            penalty_weight (float): Penalty lambda in penalty mode (default 1000.0).
            stagnation_iters (int): Stagnant iterations before velocity kick (default 15).
            use_local_search (bool): Enable memetic intensification (default False in `PSOSolver`).
            ls_operator (str): Operator "move", "kempe", "swap", or "hybrid" (default "hybrid").
            ls_top_k (int): Top particles refined per iteration in memetic mode (default 1).
            ls_max_passes (int): Move passes inside `intensify` (default 3).
            ls_kempe_steps (int): Kempe chain trials inside `intensify` (default 20).
            ls_swap_trials (int): Swap trials inside `intensify` (default 25).
        """
        defaults: dict[str, Any] = {
            "swarm_size": 20,
            "encoding": "random_keys",
            "decode_mode": "priority_greedy",
            "w_max": 0.78,
            "w_min": 0.42,
            "c1": 1.49,
            "c2": 1.49,
            "v_max": 0.35,
            "mutation_rate": 0.03,
            "dsatur_init_ratio": 0.35,
            "constraint_handling": "repair",
            "max_repair_passes": 3,
            "penalty_weight": 1000.0,
            "stagnation_iters": 15,
            "use_local_search": False,
            "ls_operator": "hybrid",
            "ls_top_k": 1,
            "ls_max_passes": 1,
            "ls_kempe_steps": 10,
            "ls_swap_trials": 10,
        }
        if hparams:
            defaults.update(hparams)
        defaults.update(kwargs)
        super().__init__(defaults)

    def _evaluate_particle(
        self,
        inst: Instance,
        cand_x: np.ndarray,
        rng: np.random.Generator,
        use_repair: bool,
        max_repair_passes: int,
        lam: float,
        budget: Budget,
    ) -> tuple[np.ndarray, float, float, float]:
        """Optionally repair `cand_x`, evaluate `(f, V, F)`, and charge 1.0 FEE."""
        if use_repair:
            eval_x = repair(inst, cand_x, rng, max_passes=max_repair_passes)
        else:
            eval_x = np.array(cand_x, dtype=np.int32, copy=True)

        v_val = violation_weight(inst, eval_x)
        f_val = proximity_cost(inst, eval_x)
        pen_f = f_val + lam * (v_val / float(inst.n_students))
        budget.charge_full(1)
        return eval_x, f_val, v_val, pen_f

    def solve(self, inst: Instance, budget: Budget, seed: int) -> Result:
        """Run Particle Swarm Optimization within `budget` using deterministic `seed`."""
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        tracker = Tracker(budget)

        n_exams = inst.n_exams
        n_slots = inst.n_slots
        swarm_size = max(4, int(self.hparams.get("swarm_size", 24)))
        encoding = str(self.hparams.get("encoding", "random_keys")).lower().strip()
        if encoding not in {"random_keys", "direct_discrete"}:
            raise ValueError(
                f"Unknown PSO encoding {encoding!r}; expected 'random_keys' or 'direct_discrete'."
            )

        decode_mode = str(self.hparams.get("decode_mode", "priority_greedy")).lower().strip()
        w_max = float(self.hparams.get("w_max", 0.78))
        w_min = float(self.hparams.get("w_min", 0.42))
        c1 = float(self.hparams.get("c1", 1.49))
        c2 = float(self.hparams.get("c2", 1.49))
        v_max = float(self.hparams.get("v_max", 0.35))
        mut_rate = float(self.hparams.get("mutation_rate", 0.03))
        dsatur_ratio = float(self.hparams.get("dsatur_init_ratio", 0.35))

        mode_str = str(self.hparams.get("constraint_handling", "repair")).lower().strip()
        use_repair = mode_str == "repair"
        max_rep_passes = max(1, int(self.hparams.get("max_repair_passes", 4)))
        lam = float(self.hparams.get("penalty_weight", 1000.0))
        stag_limit = max(4, int(self.hparams.get("stagnation_iters", 15)))

        use_ls = bool(self.hparams.get("use_local_search", False))
        ls_op = str(self.hparams.get("ls_operator", "hybrid"))
        ls_top_k = max(1, int(self.hparams.get("ls_top_k", 2)))
        ls_passes = max(1, int(self.hparams.get("ls_max_passes", 3)))
        ls_kempe = max(0, int(self.hparams.get("ls_kempe_steps", 20)))
        ls_swaps = max(0, int(self.hparams.get("ls_swap_trials", 25)))

        # Allocate swarm arrays
        pop_x = np.zeros((swarm_size, n_exams), dtype=np.int32)
        pbest_x = np.zeros((swarm_size, n_exams), dtype=np.int32)
        pbest_fit = np.full(swarm_size, float("inf"), dtype=np.float64)

        if encoding == "random_keys":
            pos_keys = rng.random((swarm_size, n_exams))
            vel_keys = rng.uniform(-0.15, 0.15, size=(swarm_size, n_exams))
            pbest_keys = pos_keys.copy()
        else:
            vel_prob = rng.uniform(0.15, 0.45, size=(swarm_size, n_exams))

        best_feasible_x: np.ndarray | None = None
        best_feasible_f: float = float("inf")
        gbest_x: np.ndarray = np.zeros(n_exams, dtype=np.int32)
        gbest_fit: float = float("inf")
        gbest_keys: np.ndarray = np.zeros(n_exams, dtype=np.float64)

        fallback_x: np.ndarray = np.zeros(n_exams, dtype=np.int32)
        best_infeasible_v: float = float("inf")

        # ------------------------------------------------------------------
        # Phase 1: Swarm initialization (hybrid DSATUR + random diversity)
        # ------------------------------------------------------------------
        n_dsatur_init = min(swarm_size, max(1, int(round(swarm_size * dsatur_ratio))))
        initialized_count = 0

        for k in range(swarm_size):
            if budget.exhausted:
                break

            if k < n_dsatur_init and use_repair:
                cand_x = dsatur_construct(
                    inst, rng, slot_selection="min_proximity", use_repair=True
                )
                v_val = violation_weight(inst, cand_x)
                f_val = proximity_cost(inst, cand_x)
                pen_f = f_val + lam * (v_val / float(inst.n_students))
                budget.charge_full(1)
                eval_x = cand_x
                if encoding == "random_keys":
                    pos_keys[k] = encode_slots_to_keys(inst, eval_x, rng, decode_mode=decode_mode)
            else:
                if encoding == "random_keys":
                    raw_x = decode_random_keys(inst, pos_keys[k], rng, decode_mode=decode_mode)
                else:
                    raw_x = rng.integers(0, n_slots, size=n_exams, dtype=np.int32)
                eval_x, f_val, v_val, pen_f = self._evaluate_particle(
                    inst, raw_x, rng, use_repair, max_rep_passes, lam, budget
                )
                if encoding == "random_keys" and use_repair:
                    pos_keys[k] = 0.7 * pos_keys[k] + 0.3 * encode_slots_to_keys(
                        inst, eval_x, rng, decode_mode=decode_mode
                    )

            pop_x[k] = eval_x
            pbest_x[k] = eval_x.copy()
            pbest_fit[k] = pen_f
            if encoding == "random_keys":
                pbest_keys[k] = pos_keys[k].copy()

            if v_val <= _EPS and f_val < best_feasible_f:
                best_feasible_f = f_val
                best_feasible_x = eval_x.copy()
            elif v_val < best_infeasible_v:
                best_infeasible_v = v_val
                fallback_x = eval_x.copy()

            if pen_f < gbest_fit:
                gbest_fit = pen_f
                gbest_x = eval_x.copy()
                if encoding == "random_keys":
                    gbest_keys = pos_keys[k].copy()

            initialized_count += 1

        # Optional initial memetic polish on gbest
        if use_ls and not budget.exhausted and initialized_count > 0:
            polished = intensify(
                inst,
                gbest_x,
                budget=budget,
                rng=rng,
                operator=ls_op,
                max_passes=ls_passes,
                kempe_steps=ls_kempe,
                swap_trials=ls_swaps,
            )
            v_pol = violation_weight(inst, polished)
            f_pol = proximity_cost(inst, polished)
            pen_pol = f_pol + lam * (v_pol / float(inst.n_students))
            if v_pol <= _EPS and f_pol < best_feasible_f:
                best_feasible_f = f_pol
                best_feasible_x = polished.copy()
            if pen_pol < gbest_fit:
                gbest_fit = pen_pol
                gbest_x = polished.copy()
                pop_x[0] = polished.copy()
                pbest_x[0] = polished.copy()
                pbest_fit[0] = pen_pol
                if encoding == "random_keys":
                    gbest_keys = encode_slots_to_keys(inst, polished, rng, decode_mode=decode_mode)
                    pbest_keys[0] = gbest_keys.copy()

        init_div = diversity(pop_x[:initialized_count]) if initialized_count >= 2 else 0.0
        tracker.record(
            iteration=0,
            best_f_so_far=best_feasible_f,
            diversity=init_div,
            force=True,
        )

        # ------------------------------------------------------------------
        # Phase 2: Main swarm evolution loop
        # ------------------------------------------------------------------
        iteration = 1
        stagnation_counter = 0

        while not budget.exhausted and initialized_count > 0:
            # Linearly decrease inertia weight w from w_max to w_min as budget is consumed
            progress = min(1.0, budget.used / max(1.0, budget.max_evals))
            w = w_max - (w_max - w_min) * progress
            improved_in_iter = False

            for k in range(swarm_size):
                if budget.exhausted:
                    break

                if encoding == "random_keys":
                    r1 = rng.random(n_exams)
                    r2 = rng.random(n_exams)
                    vel_keys[k] = (
                        w * vel_keys[k]
                        + c1 * r1 * (pbest_keys[k] - pos_keys[k])
                        + c2 * r2 * (gbest_keys - pos_keys[k])
                    )
                    vel_keys[k] = np.clip(vel_keys[k], -v_max, v_max)
                    pos_keys[k] = np.clip(pos_keys[k] + vel_keys[k], 0.0, 1.0)

                    # Apply slight Gaussian key turbulence on a small fraction of exams
                    if mut_rate > 0.0:
                        mut_mask = rng.random(n_exams) < mut_rate
                        n_mut = int(np.sum(mut_mask))
                        if n_mut > 0:
                            pos_keys[k, mut_mask] = np.clip(
                                pos_keys[k, mut_mask] + rng.normal(0.0, 0.20, size=n_mut),
                                0.0,
                                1.0,
                            )

                    cand_x = decode_random_keys(inst, pos_keys[k], rng, decode_mode=decode_mode)
                else:
                    cand_x, vel_prob[k] = update_direct_discrete_particle(
                        pop_x[k],
                        vel_prob[k],
                        pbest_x[k],
                        gbest_x,
                        n_slots=n_slots,
                        w=w,
                        c1=c1,
                        c2=c2,
                        mutation_rate=mut_rate,
                        rng=rng,
                    )

                eval_x, f_val, v_val, pen_f = self._evaluate_particle(
                    inst, cand_x, rng, use_repair, max_rep_passes, lam, budget
                )
                pop_x[k] = eval_x

                # Update personal best
                if pen_f < pbest_fit[k] - _EPS:
                    pbest_fit[k] = pen_f
                    pbest_x[k] = eval_x.copy()
                    if encoding == "random_keys":
                        if use_repair:
                            pbest_keys[k] = 0.6 * pos_keys[k] + 0.4 * encode_slots_to_keys(
                                inst, eval_x, rng, decode_mode=decode_mode
                            )
                        else:
                            pbest_keys[k] = pos_keys[k].copy()

                # Update global best and best feasible schedule
                if v_val <= _EPS and f_val < best_feasible_f - _EPS:
                    best_feasible_f = f_val
                    best_feasible_x = eval_x.copy()
                    improved_in_iter = True
                elif v_val < best_infeasible_v:
                    best_infeasible_v = v_val
                    fallback_x = eval_x.copy()

                if pen_f < gbest_fit - _EPS:
                    gbest_fit = pen_f
                    gbest_x = eval_x.copy()
                    if encoding == "random_keys":
                        gbest_keys = pbest_keys[k].copy()
                    improved_in_iter = True

            # --------------------------------------------------------------
            # Memetic Local Search intensification on top-k particles
            # --------------------------------------------------------------
            if use_ls and not budget.exhausted:
                top_indices = np.argsort(pbest_fit)[: min(ls_top_k, swarm_size)]
                for idx_raw in top_indices:
                    if budget.exhausted:
                        break
                    idx = int(idx_raw)
                    refined_x = intensify(
                        inst,
                        pbest_x[idx],
                        budget=budget,
                        rng=rng,
                        operator=ls_op,
                        max_passes=ls_passes,
                        kempe_steps=ls_kempe,
                        swap_trials=ls_swaps,
                    )
                    v_ref = violation_weight(inst, refined_x)
                    f_ref = proximity_cost(inst, refined_x)
                    pen_ref = f_ref + lam * (v_ref / float(inst.n_students))

                    if pen_ref < pbest_fit[idx] - _EPS:
                        pbest_fit[idx] = pen_ref
                        pbest_x[idx] = refined_x.copy()
                        pop_x[idx] = refined_x.copy()
                        if encoding == "random_keys":
                            pbest_keys[idx] = encode_slots_to_keys(
                                inst, refined_x, rng, decode_mode=decode_mode
                            )
                            pos_keys[idx] = pbest_keys[idx].copy()

                    if v_ref <= _EPS and f_ref < best_feasible_f - _EPS:
                        best_feasible_f = f_ref
                        best_feasible_x = refined_x.copy()
                        improved_in_iter = True

                    if pen_ref < gbest_fit - _EPS:
                        gbest_fit = pen_ref
                        gbest_x = refined_x.copy()
                        if encoding == "random_keys":
                            gbest_keys = pbest_keys[idx].copy()
                        improved_in_iter = True

            # Stagnation diversification kick on worst half of swarm
            if improved_in_iter:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                if stagnation_counter >= stag_limit and not budget.exhausted:
                    worst_half = np.argsort(-pbest_fit)[: swarm_size // 2]
                    if encoding == "random_keys":
                        pos_keys[worst_half] = rng.random((worst_half.shape[0], n_exams))
                        vel_keys[worst_half] = rng.uniform(
                            -v_max, v_max, size=(worst_half.shape[0], n_exams)
                        )
                    else:
                        vel_prob[worst_half] = rng.uniform(
                            0.35, 0.80, size=(worst_half.shape[0], n_exams)
                        )
                    stagnation_counter = 0

            div_score = diversity(pop_x) if swarm_size >= 2 else 0.0
            tracker.record(
                iteration=iteration,
                best_f_so_far=best_feasible_f,
                diversity=div_score,
            )
            iteration += 1

        # Final exact verification on best_feasible_x
        if best_feasible_x is not None:
            if violation_weight(inst, best_feasible_x) == 0.0:
                best_feasible_f = proximity_cost(inst, best_feasible_x)
            else:
                best_feasible_x = None
                best_feasible_f = float("inf")

        final_div = diversity(pop_x) if initialized_count >= 2 else 0.0
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


class PSOMemeticSolver(PSOSolver):
    """Memetic Particle Swarm Optimization solver (`pso_memetic`).

    Combines PSO swarm exploration with Lamarckian local search (`intensify`
    from `src/local_search/shared_ls.py`) applied to the top personal-best
    particles each iteration.
    """

    name: str = "pso_memetic"

    def __init__(self, hparams: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize PSOMemeticSolver with `use_local_search=True` by default."""
        defaults: dict[str, Any] = {
            "swarm_size": 20,
            "encoding": "random_keys",
            "decode_mode": "priority_greedy",
            "w_max": 0.75,
            "w_min": 0.40,
            "c1": 1.49,
            "c2": 1.49,
            "v_max": 0.35,
            "mutation_rate": 0.03,
            "dsatur_init_ratio": 0.35,
            "constraint_handling": "repair",
            "max_repair_passes": 3,
            "penalty_weight": 1000.0,
            "stagnation_iters": 12,
            "use_local_search": True,
            "ls_operator": "hybrid",
            "ls_top_k": 1,
            "ls_max_passes": 1,
            "ls_kempe_steps": 10,
            "ls_swap_trials": 10,
        }
        if hparams:
            defaults.update(hparams)
        defaults.update(kwargs)
        super().__init__(defaults)
