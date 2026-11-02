"""Contract verification tests for v0.0-contract signatures and basic invariants."""

from pathlib import Path

import numpy as np
import yaml

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


def _toy_instance() -> Instance:
    """Build a small 4-exam, 3-slot toy instance in CSR format."""
    # Edges: (0,1) w=4, (1,2) w=3, (2,3) w=2 -> |E| = 3
    indptr = np.array([0, 1, 3, 5, 6], dtype=np.int32)
    indices = np.array([1, 0, 2, 1, 3, 2], dtype=np.int32)
    weights = np.array([4, 4, 3, 3, 2, 2], dtype=np.int32)
    exam_size = np.array([6, 7, 5, 4], dtype=np.int32)
    return Instance(
        name="toy-4",
        n_exams=4,
        n_students=10,
        n_slots=3,
        indptr=indptr,
        indices=indices,
        weights=weights,
        exam_size=exam_size,
        n_edges=3,
        meta={"source": "test", "seed": 1000},
    )


def test_csr_and_evaluation() -> None:
    """Check CSR degree, feasibility, and Carter proximity cost calculation."""
    inst = _toy_instance()
    assert inst.degree(0) == 1
    assert inst.degree(1) == 2

    x_ok = np.array([0, 2, 0, 2], dtype=np.int32)
    assert is_feasible(inst, x_ok)
    assert violation_weight(inst, x_ok) == 0.0
    assert np.isclose(proximity_cost(inst, x_ok), 7.2)
    assert np.isclose(fitness(inst, x_ok), 7.2)


def test_delta_move_matches_full_eval() -> None:
    """Check that delta_move returns exact differences in f(x) and V(x)."""
    inst = _toy_instance()
    x = np.array([0, 1, 2, 0], dtype=np.int32)
    f0 = proximity_cost(inst, x)
    v0 = violation_weight(inst, x)

    for i in range(inst.n_exams):
        for s_new in range(inst.n_slots):
            df, dv = delta_move(inst, x, i, s_new)
            x2 = x.copy()
            x2[i] = s_new
            assert np.isclose(f0 + df, proximity_cost(inst, x2))
            assert np.isclose(v0 + dv, violation_weight(inst, x2))


def test_budget_and_tracker() -> None:
    """Check FEE budget accounting and Tracker history shape."""
    inst = _toy_instance()
    budget = Budget(max_evals=5.0)
    tracker = Tracker(budget, step_fraction=0.2)

    budget.charge_full(1)
    budget.charge_delta(inst, 1)  # adds 2/3 FEE
    assert np.isclose(budget.used, 1.0 + 2.0 / 3.0)

    tracker.record(iteration=0, best_f_so_far=7.2, diversity=0.5, force=True)
    hist = tracker.to_array()
    assert hist.shape == (1, 5)


def test_empty_conflict_graph() -> None:
    """Check that an instance without conflict edges is handled safely."""
    inst = Instance(
        name="empty-edges",
        n_exams=2,
        n_students=3,
        n_slots=2,
        indptr=np.array([0, 0, 0], dtype=np.int32),
        indices=np.array([], dtype=np.int32),
        weights=np.array([], dtype=np.int32),
        exam_size=np.array([1, 1], dtype=np.int32),
        n_edges=0,
        meta={"source": "test"},
    )
    x = np.array([0, 0], dtype=np.int32)
    assert is_feasible(inst, x)
    assert proximity_cost(inst, x) == 0.0
    assert violation_weight(inst, x) == 0.0
    assert delta_move(inst, x, 0, 1) == (0.0, 0.0)

    budget = Budget(max_evals=1.0)
    budget.charge_delta(inst, 0)
    assert budget.used == 0.0


def test_repair_and_yaml_files() -> None:
    """Check repair helper and presence of data/instances.yaml."""
    inst = _toy_instance()
    rng = np.random.default_rng(1000)
    x_rep = repair(inst, np.zeros(4, dtype=np.int32), rng)
    assert is_feasible(inst, x_rep)

    sol = random_solution(inst, rng)
    assert sol.shape == (4,)
    assert np.isclose(diversity(np.array([[0, 1, 2, 0], [0, 2, 2, 1]])), 0.5)

    root = Path(__file__).resolve().parent.parent
    data_cfg = yaml.safe_load((root / "data" / "instances.yaml").read_text(encoding="utf-8"))
    assert "tiny" in data_cfg["groups"]


if __name__ == "__main__":
    test_csr_and_evaluation()
    test_delta_move_matches_full_eval()
    test_budget_and_tracker()
    test_repair_and_yaml_files()
    test_empty_conflict_graph()
    print("All contract tests passed!")
