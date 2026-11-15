"""Core problem evaluation and delta-move kernels.

Implements Numba-accelerated evaluation kernels operating directly on the
CSR conflict graph, conforming to the frozen interface contract.
"""

from __future__ import annotations

import numpy as np

from src.core.instance import Instance

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - fallback when numba is unavailable
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def _decorator(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return _decorator


@njit(cache=True)
def _slot_distance_weight(d: int) -> int:
    """Return Carter proximity weight w(d) for slot distance d = |x_i - x_j|."""
    # Carter penalty applies only when two conflicting exams are 1..5 slots apart
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
    # Scan upper triangle (j > i) of the CSR adjacency list to count each edge once
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
    # Sum shared student counts for all conflicting pairs assigned to the same slot
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
    # Moving to the same slot causes zero change in both objective and violations
    if s_old == s_new:
        return 0.0, 0.0

    delta_raw_f = 0
    delta_v = 0
    start = indptr[exam_i]
    end = indptr[exam_i + 1]

    # Visit only the conflicting neighbours of exam_i in O(deg(i)) time
    for k in range(start, end):
        j = int(indices[k])
        w_ij = int(weights[k])
        s_j = int(x[j])

        # Subtract contribution of the old slot assignment
        d_old = abs(s_old - s_j)
        if d_old == 0:
            delta_v -= w_ij
        elif d_old <= 5:
            delta_raw_f -= w_ij * (1 << (5 - d_old))

        # Add contribution of the candidate new slot assignment
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
    # Compare all unique pairs of solutions across all N exam positions
    for a in range(n_sols - 1):
        for b in range(a + 1, n_sols):
            for i in range(n_exams):
                if X[a, i] != X[b, i]:
                    diff_count += 1

    n_pairs = (n_sols * (n_sols - 1)) // 2
    return float(diff_count) / float(n_pairs * n_exams)


def get_proximity_kernel(inst: Instance) -> np.ndarray:
    """Return cached (T, T) Carter proximity kernel K[s, r] = 2^(5-|s-r|) for 1<=|s-r|<=5."""
    if not hasattr(inst, "_k_prox"):
        s_grid = np.arange(inst.n_slots)
        d_mat = np.abs(s_grid[:, None] - s_grid[None, :])
        k_prox = np.where((d_mat >= 1) & (d_mat <= 5), 1 << np.clip(5 - d_mat, 0, 5), 0).astype(
            np.float64
        )
        object.__setattr__(inst, "_k_prox", k_prox)
    return inst._k_prox


def _ensure_upper_edges(inst: Instance) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache upper-triangle edge arrays (j > i) on inst for fast vectorized fallback."""
    if not hasattr(inst, "_u_upper"):
        deg = np.diff(inst.indptr)
        u_all = np.repeat(np.arange(inst.n_exams, dtype=np.int32), deg)
        mask = inst.indices > u_all
        object.__setattr__(inst, "_u_upper", u_all[mask])
        object.__setattr__(inst, "_v_upper", inst.indices[mask])
        object.__setattr__(inst, "_w_upper", inst.weights[mask])
    return inst._u_upper, inst._v_upper, inst._w_upper


def proximity_cost(inst: Instance, x: np.ndarray) -> float:
    """Compute the Carter proximity cost f(x) of a timetable.

    Args:
        inst: Problem instance in CSR format.
        x: Slot assignment vector of shape (N,).

    Returns:
        Normalized Carter proximity cost per student. Time complexity: O(|E|).
    """
    if _HAS_NUMBA:
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
    u, v, w = _ensure_upper_edges(inst)
    if u.shape[0] == 0:
        return 0.0
    d = np.abs(x[u] - x[v])
    mask = (d >= 1) & (d <= 5)
    if not np.any(mask):
        return 0.0
    total = np.sum(w[mask] * (1 << (5 - d[mask])))
    return float(total) / float(inst.n_students)


def violation_weight(inst: Instance, x: np.ndarray) -> float:
    """Compute total conflicting student enrolments V(x) for same-slot exams.

    Args:
        inst: Problem instance in CSR format.
        x: Slot assignment vector of shape (N,).

    Returns:
        Total weight of hard conflicts. Time complexity: O(|E|).
    """
    if _HAS_NUMBA:
        return float(
            _violation_weight_kernel(
                inst.n_exams,
                inst.indptr,
                inst.indices,
                inst.weights,
                x,
            )
        )
    u, v, w = _ensure_upper_edges(inst)
    if u.shape[0] == 0:
        return 0.0
    return float(np.sum(w[x[u] == x[v]]))


def fitness(inst: Instance, x: np.ndarray, lam: float = 1000.0) -> float:
    """Compute penalized objective F(x) = f(x) + lam * V(x) / S.

    Args:
        inst: Problem instance in CSR format.
        x: Slot assignment vector of shape (N,).
        lam: Penalty multiplier for hard constraint violations.

    Returns:
        Generalized penalized fitness value. Time complexity: O(|E|).
    """
    f_val = proximity_cost(inst, x)
    v_val = violation_weight(inst, x)
    return f_val + float(lam) * (v_val / float(inst.n_students))


def delta_move(inst: Instance, x: np.ndarray, i: int, s_new: int) -> tuple[float, float]:
    """Compute incremental change (delta_f, delta_V) when moving exam `i` to `s_new`.

    Does not modify `x` in place.

    Args:
        inst: Problem instance in CSR format.
        x: Current slot assignment vector of shape (N,).
        i: Index of the exam being moved.
        s_new: Candidate target slot index in [0, T - 1].

    Returns:
        Tuple `(delta_f, delta_V)`. Time complexity: O(deg(i)).
    """
    if _HAS_NUMBA:
        return _delta_move_kernel(
            inst.n_students,
            inst.indptr,
            inst.indices,
            inst.weights,
            x,
            int(i),
            int(s_new),
        )
    s_old = int(x[i])
    s_cand = int(s_new)
    if s_old == s_cand:
        return 0.0, 0.0
    start = int(inst.indptr[i])
    end = int(inst.indptr[i + 1])
    if start == end:
        return 0.0, 0.0
    nbrs = inst.indices[start:end]
    wts = inst.weights[start:end]
    s_j = x[nbrs]

    d_old = np.abs(s_old - s_j)
    v_old = int(np.sum(wts[d_old == 0]))
    m_old = (d_old >= 1) & (d_old <= 5)
    f_old = int(np.sum(wts[m_old] * (1 << (5 - d_old[m_old])))) if np.any(m_old) else 0

    d_new = np.abs(s_cand - s_j)
    v_new = int(np.sum(wts[d_new == 0]))
    m_new = (d_new >= 1) & (d_new <= 5)
    f_new = int(np.sum(wts[m_new] * (1 << (5 - d_new[m_new])))) if np.any(m_new) else 0

    return float(f_new - f_old) / float(inst.n_students), float(v_new - v_old)


def is_feasible(inst: Instance, x: np.ndarray) -> bool:
    """Check whether timetable `x` satisfies all hard conflict constraints.

    Args:
        inst: Problem instance in CSR format.
        x: Slot assignment vector of shape (N,).

    Returns:
        True if all slots are in [0, T - 1] and V(x) == 0. Time complexity: O(|E|).
    """
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
    """Greedy randomized conflict repair with incremental slot-conflict table.

    Maintains `slot_table[i, s]` storing the conflict weight of exam `i` with
    neighbours currently assigned to slot `s`. In each pass, conflicted exams
    are moved to conflict-minimizing slots (breaking ties by Carter proximity
    cost), followed by a short Tabu-assisted min-conflicts phase if any hard
    conflicts persist.

    Args:
        inst: Problem instance in CSR format.
        x: Initial timetable vector (copied before modification).
        rng: Isolated NumPy random generator.
        max_passes: Maximum sweeps over conflicted exams.

    Returns:
        Repaired timetable array of shape (N,) and dtype int32.
    """
    repaired = np.array(x, dtype=np.int32, copy=True)
    n_exams = inst.n_exams
    n_slots = inst.n_slots
    if n_slots <= 1 or max_passes <= 0:
        return repaired

    indptr = inst.indptr
    indices = inst.indices
    weights = inst.weights

    # Build initial slot conflict table C[i, s] via cached BLAS adjacency product
    if not hasattr(inst, "_adj_f32"):
        adj = np.zeros((n_exams, n_exams), dtype=np.float32)
        deg = np.diff(indptr)
        u_all = np.repeat(np.arange(n_exams, dtype=np.int32), deg)
        adj[u_all, indices] = weights.astype(np.float32)
        object.__setattr__(inst, "_adj_f32", adj)
    one_hot = np.zeros((n_exams, n_slots), dtype=np.float32)
    one_hot[np.arange(n_exams), repaired] = 1.0
    slot_table = np.rint(inst._adj_f32 @ one_hot).astype(np.int32)

    exam_idx = np.arange(n_exams)

    # Phase 1: Greedy conflict reduction passes
    for _ in range(max_passes):
        conf_exams = np.flatnonzero(slot_table[exam_idx, repaired] > 0)
        if conf_exams.shape[0] == 0:
            return repaired

        improved_in_pass = False
        for u_raw in rng.permutation(conf_exams):
            u = int(u_raw)
            s_cur = int(repaired[u])
            c_cur = int(slot_table[u, s_cur])
            if c_cur == 0:
                continue

            min_c = int(np.min(slot_table[u]))
            if min_c < c_cur:
                cands = np.flatnonzero(slot_table[u] == min_c)
                if cands.shape[0] == 1:
                    s_new = int(cands[0])
                else:
                    k_prox = get_proximity_kernel(inst)
                    scores = (k_prox[cands] @ slot_table[u]) + rng.random(cands.shape[0]) * 1e-3
                    s_new = int(cands[int(np.argmin(scores))])

                nbrs = indices[indptr[u] : indptr[u + 1]]
                wts = weights[indptr[u] : indptr[u + 1]]
                slot_table[nbrs, s_cur] -= wts
                slot_table[nbrs, s_new] += wts
                repaired[u] = s_new
                improved_in_pass = True

        if not improved_in_pass:
            break

    # Phase 2: Short Tabu-assisted min-conflicts search if conflicts remain
    cur_v = int(np.sum(slot_table[exam_idx, repaired]) // 2)
    if cur_v > 0:
        best_x = repaired.copy()
        best_v = cur_v
        tabu_until = np.zeros((n_exams, n_slots), dtype=np.int32)
        max_tabu_steps = min(150, max_passes * 30) if n_exams <= 200 else 15

        for step in range(1, max_tabu_steps + 1):
            conf_exams = np.flatnonzero(slot_table[exam_idx, repaired] > 0)
            if conf_exams.shape[0] == 0:
                return repaired

            cur_slots = repaired[conf_exams]
            cur_confs = slot_table[conf_exams, cur_slots][:, None]
            delta_mat = slot_table[conf_exams] - cur_confs
            delta_mat[np.arange(conf_exams.shape[0]), cur_slots] = 10**9

            is_tabu = tabu_until[conf_exams] >= step
            aspiration = (cur_v + delta_mat) < best_v
            eff_delta = np.where((~is_tabu) | aspiration, delta_mat, 10**8 + delta_mat)

            min_d = int(np.min(eff_delta))
            best_pairs = np.argwhere(eff_delta == min_d)
            pick = int(rng.integers(0, best_pairs.shape[0]))
            row_i, s_new = int(best_pairs[pick, 0]), int(best_pairs[pick, 1])
            u = int(conf_exams[row_i])
            s_old = int(repaired[u])
            dv = int(delta_mat[row_i, s_new])

            nbrs = indices[indptr[u] : indptr[u + 1]]
            wts = weights[indptr[u] : indptr[u + 1]]
            slot_table[nbrs, s_old] -= wts
            slot_table[nbrs, s_new] += wts
            repaired[u] = s_new
            cur_v += dv
            tabu_until[u, s_old] = step + int(0.6 * conf_exams.shape[0]) + int(rng.integers(1, 7))

            if cur_v < best_v:
                best_v = cur_v
                best_x = repaired.copy()
                if best_v == 0:
                    return best_x

        return best_x

    return repaired


def random_solution(inst: Instance, rng: np.random.Generator) -> np.ndarray:
    """Generate a random timetable and apply the greedy repair operator.

    Args:
        inst: Problem instance in CSR format.
        rng: Isolated NumPy random generator.

    Returns:
        Timetable array of shape (N,) and dtype int32.
    """
    x = rng.integers(0, inst.n_slots, size=inst.n_exams, dtype=np.int32)
    return repair(inst, x, rng)


def diversity(X: np.ndarray) -> float:
    """Compute mean normalized pairwise Hamming distance across a population.

    Args:
        X: 2D array of shape (K, N) containing K candidate timetables.

    Returns:
        Normalized diversity in [0.0, 1.0]. Time complexity: O(K^2 * N).
    """
    arr = np.asarray(X, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D population array (K, N), got {arr.ndim}D.")
    if _HAS_NUMBA:
        return float(_diversity_kernel(arr))
    n_sols, n_ex = arr.shape
    if n_sols <= 1 or n_ex == 0:
        return 0.0
    diffs = np.sum(arr[:, None, :] != arr[None, :, :], axis=2)
    return float(np.sum(np.triu(diffs, k=1))) / float((n_sols * (n_sols - 1) // 2) * n_ex)
