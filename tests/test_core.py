"""Unit tests for P3 Core Framework (`instance`, `problem`, `solver`, `tracker`).

Verifies CSR invariants, exact match between `delta_move` and full evaluation on
generated benchmark instances, repair feasibility, diversity bounds, FEE budget
accounting, and tracker logging intervals.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.instance import Instance
from src.core.problem import (
    delta_move,
    diversity,
    fitness,
    is_feasible,
    proximity_cost,
    random_solution,
    repair,
    violation_weight,
)
from src.core.solver import Budget
from src.core.tracker import Tracker
from src.data import load_instance


def test_instance_validation_and_immutability() -> None:
    """Verify that `Instance` validates CSR shapes and locks numpy buffers."""
    indptr = np.array([0, 1, 2], dtype=np.int32)
    indices = np.array([1, 0], dtype=np.int32)
    weights = np.array([5, 5], dtype=np.int32)
    exam_size = np.array([8, 9], dtype=np.int32)

    inst = Instance(
        name="pair-2",
        n_exams=2,
        n_students=12,
        n_slots=3,
        indptr=indptr,
        indices=indices,
        weights=weights,
        exam_size=exam_size,
        n_edges=1,
    )
    assert inst.degree(0) == 1
    assert inst.neighbors(0)[0].tolist() == [1]
    assert not inst.indptr.flags.writeable

    # Bad indptr length must raise ValueError
    with pytest.raises(ValueError, match="indptr length"):
        Instance(
            name="bad",
            n_exams=3,
            n_students=10,
            n_slots=2,
            indptr=indptr,
            indices=indices,
            weights=weights,
            exam_size=np.array([1, 1, 1], dtype=np.int32),
            n_edges=1,
        )


def test_delta_move_exact_on_generated_benchmarks() -> None:
    """Verify delta_move matches full evaluation across all exams on tiny-a and small-a."""
    rng = np.random.default_rng(42)
    for name in ("tiny-a", "small-a"):
        inst = load_instance(name)
        x = rng.integers(0, inst.n_slots, size=inst.n_exams, dtype=np.int32)
        base_f = proximity_cost(inst, x)
        base_v = violation_weight(inst, x)
        x_snapshot = x.copy()

        for exam_i in range(inst.n_exams):
            for s_new in range(inst.n_slots):
                df, dv = delta_move(inst, x, exam_i, s_new)
                # Ensure delta_move never mutates x in place
                assert np.array_equal(x, x_snapshot)

                x_Moved = x.copy()
                x_Moved[exam_i] = s_new
                assert np.isclose(base_f + df, proximity_cost(inst, x_Moved), atol=1e-9)
                assert np.isclose(base_v + dv, violation_weight(inst, x_Moved), atol=1e-9)


def test_repair_and_random_solution_feasibility() -> None:
    """Verify `repair` and `random_solution` eliminate conflicts on `tiny` and `small` instances."""
    rng = np.random.default_rng(1000)
    for name in ("tiny-a", "tiny-b", "small-a"):
        inst = load_instance(name)
        # Start from an all-zero maximally conflicted assignment
        x_bad = np.zeros(inst.n_exams, dtype=np.int32)
        assert not is_feasible(inst, x_bad)

        x_rep = repair(inst, x_bad, rng, max_passes=5)
        assert is_feasible(inst, x_rep)
        assert violation_weight(inst, x_rep) == 0.0
        assert np.isclose(fitness(inst, x_rep), proximity_cost(inst, x_rep))

        x_rand = random_solution(inst, rng)
        assert x_rand.shape == (inst.n_exams,)
        assert is_feasible(inst, x_rand)


def test_diversity_invariants() -> None:
    """Check `diversity` on identical, complementary, and invalid population arrays."""
    identical = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32)
    assert diversity(identical) == 0.0

    distinct = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int32)
    assert np.isclose(diversity(distinct), 1.0)

    single = np.array([[0, 1, 2, 3]], dtype=np.int32)
    assert diversity(single) == 0.0

    with pytest.raises(ValueError, match="2D"):
        diversity(np.array([0, 1, 2], dtype=np.int32))


def test_budget_and_tracker_checkpoints() -> None:
    """Verify FEE budget charging, exhaustion, and Tracker 0.5% interval snapshots."""
    budget = Budget(max_evals=100.0)
    tracker = Tracker(budget, step_fraction=0.10)  # every 10 FEE

    assert not budget.exhausted
    for it in range(10):
        budget.charge_full(10)
        tracker.record(iteration=it, best_f_so_far=25.0 - it * 0.5, diversity=0.4)

    assert budget.exhausted
    assert np.isclose(budget.used, 100.0)
    hist = tracker.to_array()
    assert hist.ndim == 2 and hist.shape[1] == 5
    assert hist.shape[0] == 10
    assert np.all(np.diff(hist[:, 0]) >= 0.0)
