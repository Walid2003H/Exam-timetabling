"""Core problem evaluation and delta-move kernels.

Implements Numba-accelerated evaluation kernels operating directly on the
CSR conflict graph, conforming to the frozen interface contract.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from src.core.instance import Instance


@njit(cache=True)
def _slot_distance_weight(d: int) -> int:
    """Return Carter proximity weight w(d) for slot distance d = |x_i - x_j|."""
    if 1 <= d <= 5:
        return 1 << (5 - d)
    return 0


@njit(cache=True)
def _proximity_cost_kernel(
    n_exams: int,
    n_students: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    x: np.ndarray,
) -> float:
    """Numba kernel for computing Carter proximity cost f(x)."""
    total_weighted_cost = 0
    for i in range(n_exams):
        s_i = int(x[i])
        start = indptr[i]
        end = indptr[i + 1]
        for k in range(start, end):
            j = int(indices[k])
            if j > i:
                d = abs(s_i - int(x[j]))
                if 1 <= d <= 5:
                    total_weighted_cost += int(weights[k]) * (1 << (5 - d))
    return float(total_weighted_cost) / float(n_students)


@njit(cache=True)
def _violation_weight_kernel(
    n_exams: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    x: np.ndarray,
) -> float:
    """Numba kernel for computing hard constraint violation weight V(x)."""
    total_violations = 0
    for i in range(n_exams):
        s_i = int(x[i])
        start = indptr[i]
        end = indptr[i + 1]
        for k in range(start, end):
            j = int(indices[k])
            if j > i and s_i == int(x[j]):
                total_violations += int(weights[k])
    return float(total_violations)


@njit(cache=True)
def _delta_move_kernel(
    n_students: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    x: np.ndarray,
    exam_i: int,
    s_new: int,
) -> tuple[float, float]:
    """Numba kernel for O(deg(i)) incremental evaluation of moving `exam_i`."""
    s_old = int(x[exam_i])
    if s_old == s_new:
        return 0.0, 0.0

    delta_raw_f = 0
    delta_v = 0
    start = indptr[exam_i]
    end = indptr[exam_i + 1]

    for k in range(start, end):
        j = int(indices[k])
        w_ij = int(weights[k])
        s_j = int(x[j])

        d_old = abs(s_old - s_j)
        if d_old == 0:
            delta_v -= w_ij
        elif d_old <= 5:
            delta_raw_f -= w_ij * (1 << (5 - d_old))

        d_new = abs(s_new - s_j)
        if d_new == 0:
            delta_v += w_ij
        elif d_new <= 5:
            delta_raw_f += w_ij * (1 << (5 - d_new))

    return float(delta_raw_f) / float(n_students), float(delta_v)


@njit(cache=True)
def _exam_conflict_weight_kernel(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    x: np.ndarray,
    exam_i: int,
) -> int:
    """Return total conflict weight involving `exam_i` at its current slot."""
    s_i = int(x[exam_i])
    c_sum = 0
    for k in range(indptr[exam_i], indptr[exam_i + 1]):
        if int(x[indices[k]]) == s_i:
            c_sum += int(weights[k])
    return c_sum


@njit(cache=True)
def _diversity_kernel(X: np.ndarray) -> float:
    """Numba kernel for mean normalized pairwise Hamming distance."""
    n_sols, n_exams = X.shape
    if n_sols <= 1 or n_exams == 0:
        return 0.0

    diff_count = 0
    for a in range(n_sols - 1):
        for b in range(a + 1, n_sols):
            for i in range(n_exams):
                if X[a, i] != X[b, i]:
                    diff_count += 1

    n_pairs = (n_sols * (n_sols - 1)) // 2
    return float(diff_count) / float(n_pairs * n_exams)


def proximity_cost(inst: Instance, x: np.ndarray) -> float:
    """Compute the Carter proximity cost f(x) of a timetable."""
    return float(
        _proximity_cost_kernel(
            inst.n_exams,
            inst.n_students,
            inst.indptr,
            inst.indices,
            inst.weights,
            x,
        )
    )


def violation_weight(inst: Instance, x: np.ndarray) -> float:
    """Compute total conflicting student enrolments V(x) for same-slot exams."""
    return float(
        _violation_weight_kernel(
            inst.n_exams,
            inst.indptr,
            inst.indices,
            inst.weights,
            x,
        )
    )


def fitness(inst: Instance, x: np.ndarray, lam: float = 1000.0) -> float:
    """Compute penalized objective F(x) = f(x) + lam * V(x) / S."""
    f_val = proximity_cost(inst, x)
    v_val = violation_weight(inst, x)
    return f_val + float(lam) * (v_val / float(inst.n_students))


def delta_move(
    inst: Instance, x: np.ndarray, i: int, s_new: int
) -> tuple[float, float]:
    """Compute incremental change (delta_f, delta_V) when moving exam `i` to `s_new`."""
    return _delta_move_kernel(
        inst.n_students,
        inst.indptr,
        inst.indices,
        inst.weights,
        x,
        int(i),
        int(s_new),
    )


def is_feasible(inst: Instance, x: np.ndarray) -> bool:
    """Check whether timetable `x` satisfies all hard conflict constraints."""
    if x.shape[0] != inst.n_exams:
        return False
    if np.any((x < 0) | (x >= inst.n_slots)):
        return False
    return violation_weight(inst, x) == 0.0


def repair(
    inst: Instance,
    x: np.ndarray,
    rng: np.random.Generator,
    max_passes: int = 5,
) -> np.ndarray:
    """Greedy randomized conflict repair operator."""
    repaired = np.array(x, dtype=np.int32, copy=True)
    if inst.n_slots <= 1:
        return repaired

    for _ in range(max_passes):
        conflicted = [
            exam_i
            for exam_i in range(inst.n_exams)
            if _exam_conflict_weight_kernel(
                inst.indptr, inst.indices, inst.weights, repaired, exam_i
            )
            > 0
        ]
        if not conflicted:
            break

        order = rng.permutation(np.asarray(conflicted, dtype=np.int32))
        improved_in_pass = False

        for exam_i_raw in order:
            exam_i = int(exam_i_raw)
            if (
                _exam_conflict_weight_kernel(
                    inst.indptr, inst.indices, inst.weights, repaired, exam_i
                )
                == 0
            ):
                continue

            current_slot = int(repaired[exam_i])
            best_slot = current_slot
            best_dv = 0.0
            best_df = float("inf")

            slot_order = rng.permutation(inst.n_slots)
            for s_cand_raw in slot_order:
                s_cand = int(s_cand_raw)
                if s_cand == current_slot:
                    continue
                df, dv = delta_move(inst, repaired, exam_i, s_cand)
                if dv < best_dv or (dv == best_dv and dv < 0.0 and df < best_df):
                    best_dv = dv
                    best_df = df
                    best_slot = s_cand

            if best_slot != current_slot:
                repaired[exam_i] = best_slot
                improved_in_pass = True

        if not improved_in_pass:
            break

    return repaired


def random_solution(inst: Instance, rng: np.random.Generator) -> np.ndarray:
    """Generate a random timetable and apply the greedy repair operator."""
    x = rng.integers(0, inst.n_slots, size=inst.n_exams, dtype=np.int32)
    return repair(inst, x, rng)


def diversity(X: np.ndarray) -> float:
    """Compute mean normalized pairwise Hamming distance across a population."""
    arr = np.asarray(X, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D population array (K, N), got {arr.ndim}D.")
    return float(_diversity_kernel(arr))