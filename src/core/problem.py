"""Problem evaluation functions and contract stubs (v0.0-contract).

These functions define the frozen signatures from Section 3.2 with lightweight
reference logic so P4-P8 can develop against the contract immediately while P3
implements the optimized Numba kernels in v0.1-core.
"""

import numpy as np

from src.core.instance import Instance

# Carter proximity weights for slot distance d = 1..5: w(d) = 2^(5-d)
_WEIGHTS = np.array([0.0, 16.0, 8.0, 4.0, 2.0, 1.0], dtype=np.float64)


def proximity_cost(inst: Instance, x: np.ndarray) -> float:
    """Compute Carter proximity cost f(x) for a given slot assignment x.

    Args:
        inst: Problem Instance in CSR form.
        x: Integer array of length n_exams with slot indices in [0, n_slots - 1].

    Returns:
        Normalized proximity cost f(x) per student. Time complexity: O(|E|).
    """
    cost = 0.0
    for i in range(inst.n_exams):
        s_i = int(x[i])
        for k in range(int(inst.indptr[i]), int(inst.indptr[i + 1])):
            j = int(inst.indices[k])
            if j > i:
                d = abs(s_i - int(x[j]))
                if 1 <= d <= 5:
                    cost += float(inst.weights[k]) * _WEIGHTS[d]
    return cost / float(max(1, inst.n_students))


def violation_weight(inst: Instance, x: np.ndarray) -> float:
    """Compute total hard conflict weight V(x) for exams placed in the same slot.

    Args:
        inst: Problem Instance in CSR form.
        x: Integer array of length n_exams.

    Returns:
        Sum of c_ij over all conflicting pairs (i < j) with x[i] == x[j]. Time complexity: O(|E|).
    """
    v_total = 0.0
    for i in range(inst.n_exams):
        s_i = int(x[i])
        for k in range(int(inst.indptr[i]), int(inst.indptr[i + 1])):
            j = int(inst.indices[k])
            if j > i and s_i == int(x[j]):
                v_total += float(inst.weights[k])
    return v_total


def fitness(inst: Instance, x: np.ndarray, lam: float = 1000.0) -> float:
    """Compute penalized objective F(x) = f(x) + lam * V(x) / S.

    Args:
        inst: Problem Instance.
        x: Integer array of length n_exams.
        lam: Penalty coefficient for hard constraint violations (default 1000.0).

    Returns:
        Combined penalized fitness F(x). Time complexity: O(|E|).
    """
    return proximity_cost(inst, x) + lam * violation_weight(inst, x) / float(
        max(1, inst.n_students)
    )


def delta_move(inst: Instance, x: np.ndarray, i: int, s_new: int) -> tuple[float, float]:
    """Compute incremental change (delta_f, delta_V) if exam i is moved to slot s_new.

    Args:
        inst: Problem Instance.
        x: Current solution vector of length n_exams (not modified in place).
        i: Exam index to move.
        s_new: Target time slot index.

    Returns:
        Tuple (delta_f, delta_V). Time complexity: O(deg(i)).
    """
    s_old = int(x[i])
    if s_old == s_new:
        return 0.0, 0.0

    df = 0.0
    dv = 0.0
    for k in range(int(inst.indptr[i]), int(inst.indptr[i + 1])):
        j = int(inst.indices[k])
        w_ij = float(inst.weights[k])
        s_j = int(x[j])

        d_old = abs(s_old - s_j)
        if d_old == 0:
            dv -= w_ij
        elif 1 <= d_old <= 5:
            df -= w_ij * _WEIGHTS[d_old]

        d_new = abs(s_new - s_j)
        if d_new == 0:
            dv += w_ij
        elif 1 <= d_new <= 5:
            df += w_ij * _WEIGHTS[d_new]

    return df / float(max(1, inst.n_students)), dv


def is_feasible(inst: Instance, x: np.ndarray) -> bool:
    """Return True if all slots are within [0, T-1] and V(x) == 0. Time complexity: O(N + |E|)."""
    if x.shape[0] != inst.n_exams:
        return False
    if np.any(x < 0) or np.any(x >= inst.n_slots):
        return False
    return violation_weight(inst, x) == 0.0


def repair(
    inst: Instance,
    x: np.ndarray,
    rng: np.random.Generator,
    max_passes: int = 5,
) -> np.ndarray:
    """Repair hard conflicts using randomized greedy slot reassignment.

    This is the reference implementation for the initial contract stage; P3 may
    optimize it while preserving the function signature and behavior contract.

    Args:
        inst: Problem Instance.
        x: Candidate solution vector.
        rng: NumPy random Generator created with np.random.default_rng(seed).
        max_passes: Maximum number of repair sweeps.

    Returns:
        Repaired copy of the solution vector. Time complexity: O(max_passes * |E| * T).
    """
    out = np.array(x, dtype=np.int32, copy=True)
    for _ in range(max_passes):
        changed = False
        for i in rng.permutation(inst.n_exams):
            exam_i = int(i)
            start, end = int(inst.indptr[exam_i]), int(inst.indptr[exam_i + 1])
            if not any(out[exam_i] == out[int(inst.indices[k])] for k in range(start, end)):
                continue
            best_s = int(out[exam_i])
            best_dv = 0.0
            best_df = 0.0
            for s_cand in rng.permutation(inst.n_slots):
                df, dv = delta_move(inst, out, exam_i, int(s_cand))
                if dv < best_dv or (dv == best_dv and df < best_df):
                    best_dv = dv
                    best_df = df
                    best_s = int(s_cand)
            if best_s != int(out[exam_i]):
                out[exam_i] = best_s
                changed = True
        if not changed:
            break
    return out


def random_solution(inst: Instance, rng: np.random.Generator) -> np.ndarray:
    """Generate a random solution followed by repair. Time complexity: O(N + |E| * T)."""
    x = rng.integers(0, inst.n_slots, size=inst.n_exams, dtype=np.int32)
    return repair(inst, x, rng)


def diversity(X: np.ndarray) -> float:
    """Compute mean normalized pairwise Hamming distance for population matrix X of shape (P, N)."""
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] == 0:
        return 0.0
    p_size = X.shape[0]
    total = 0.0
    pairs = 0
    for a in range(p_size):
        for b in range(a + 1, p_size):
            total += float(np.mean(X[a] != X[b]))
            pairs += 1
    return total / float(max(1, pairs))
