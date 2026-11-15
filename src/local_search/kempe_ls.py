"""Kempe-chain neighborhood local search operator (`kempe_ls`).

A Kempe chain for a seed exam `i` in slot `s_1` and target slot `s_2` is the
connected component containing `i` in the subgraph induced by exams currently
assigned to `s_1` or `s_2`. Because all internal edges within the chain swap
both endpoints between `s_1` and `s_2`, their pairwise slot distances remain
unchanged; only boundary edges connecting a chain exam to an outside exam change
distance (Thompson & Dowsland, 1998).
"""

from __future__ import annotations

import numpy as np

from src.core.instance import Instance
from src.core.solver import Budget

_EPS = 1e-9


def extract_kempe_chain(
    inst: Instance,
    x: np.ndarray,
    seed_exam: int,
    target_slot: int,
) -> list[int]:
    """Find the connected component of `seed_exam` in slots `{x[seed_exam], target_slot}`.

    Uses breadth-first search (BFS) over the CSR conflict graph restricted to
    vertices currently assigned to either `x[seed_exam]` or `target_slot`.

    Args:
        inst: Problem instance in CSR format.
        x: Current timetable vector of shape (N,).
        seed_exam: Starting exam index.
        target_slot: Alternate time slot index.

    Returns:
        List of exam indices belonging to the Kempe chain.

    Time Complexity:
        O(|K| * avg_deg) where |K| is the chain size.
    """
    source_slot = int(x[seed_exam])
    if source_slot == target_slot:
        return [int(seed_exam)]

    in_chain = np.zeros(inst.n_exams, dtype=bool)
    in_chain[seed_exam] = True
    chain: list[int] = [int(seed_exam)]
    indptr = inst.indptr
    indices = inst.indices

    head = 0
    while head < len(chain):
        u = chain[head]
        head += 1
        nbrs = indices[indptr[u] : indptr[u + 1]]
        for v_raw in nbrs:
            v = int(v_raw)
            if not in_chain[v]:
                s_v = int(x[v])
                if s_v == source_slot or s_v == target_slot:
                    in_chain[v] = True
                    chain.append(v)

    return chain


def apply_kempe_chain_step(
    inst: Instance,
    x: np.ndarray,
    seed_exam: int,
    target_slot: int,
    budget: Budget | None = None,
    accept_non_improving: bool = False,
) -> tuple[bool, float, float]:
    """Evaluate and optionally apply a Kempe-chain swap around `(seed_exam, target_slot)`.

    Computes exact `(delta_f, delta_V)` by scanning boundary edges of the chain
    and charges `deg(u) / |E|` FEE for each vertex `u` in the chain.

    Args:
        inst: Problem instance in CSR format.
        x: Timetable array modified in place if accepted.
        seed_exam: Starting exam index.
        target_slot: Alternate slot index in [0, T - 1].
        budget: Optional FEE budget tracker.
        accept_non_improving: If True, applies the swap even if non-improving.

    Returns:
        Tuple `(accepted, delta_f, delta_V)`.
    """
    s1 = int(x[seed_exam])
    s2 = int(target_slot)
    if s1 == s2:
        return False, 0.0, 0.0

    chain = extract_kempe_chain(inst, x, seed_exam, s2)
    if budget is not None and inst.n_edges > 0:
        chain_cost = sum(float(inst.degree(u)) for u in chain) / float(inst.n_edges)
        if not budget.can_afford(chain_cost):
            return False, 0.0, 0.0

    in_chain = np.zeros(inst.n_exams, dtype=bool)
    in_chain[chain] = True

    indptr = inst.indptr
    indices = inst.indices
    weights = inst.weights

    delta_raw_f = 0
    delta_v = 0

    # Evaluate boundary edges (u in chain, v not in chain)
    for u in chain:
        if budget is not None:
            budget.charge_delta(inst, u)
        s_old = int(x[u])
        s_new = s2 if s_old == s1 else s1
        start = int(indptr[u])
        end = int(indptr[u + 1])
        for k in range(start, end):
            v = int(indices[k])
            if in_chain[v]:
                continue
            w = int(weights[k])
            s_v = int(x[v])

            d_old = abs(s_old - s_v)
            if d_old == 0:
                delta_v -= w
            elif d_old <= 5:
                delta_raw_f -= w * (1 << (5 - d_old))

            d_new = abs(s_new - s_v)
            if d_new == 0:
                delta_v += w
            elif d_new <= 5:
                delta_raw_f += w * (1 << (5 - d_new))

    df = float(delta_raw_f) / float(inst.n_students)
    dv = float(delta_v)

    is_improving = (dv < -_EPS) or (abs(dv) <= _EPS and df < -_EPS)
    if is_improving or accept_non_improving:
        for u in chain:
            x[u] = s2 if int(x[u]) == s1 else s1
        return True, df, dv

    return False, df, dv


def kempe_ls(
    inst: Instance,
    x: np.ndarray,
    budget: Budget | None = None,
    rng: np.random.Generator | None = None,
    max_steps: int = 100,
) -> tuple[np.ndarray, float, float]:
    """Run stochastic first-improvement Kempe-chain local search.

    Args:
        inst: Problem instance in CSR format.
        x: Starting timetable of shape (N,).
        budget: Optional FEE budget tracker.
        rng: Isolated random generator.
        max_steps: Number of candidate Kempe chains to test.

    Returns:
        Tuple `(improved_x, total_delta_f, total_delta_v)`.

    Time Complexity:
        O(max_steps * |K| * avg_deg).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    current_x = np.array(x, dtype=np.int32, copy=True)
    total_df = 0.0
    total_dv = 0.0

    if inst.n_slots <= 1 or max_steps <= 0:
        return current_x, total_df, total_dv

    for _ in range(max_steps):
        if budget is not None and budget.exhausted:
            break

        seed_exam = int(rng.integers(0, inst.n_exams))
        cur_slot = int(current_x[seed_exam])
        target_slot = int(rng.integers(0, inst.n_slots - 1))
        if target_slot >= cur_slot:
            target_slot += 1

        accepted, df, dv = apply_kempe_chain_step(
            inst, current_x, seed_exam, target_slot, budget=budget
        )
        if accepted:
            total_df += df
            total_dv += dv

    return current_x, total_df, total_dv
