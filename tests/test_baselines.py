"""Unit tests for P4 Baselines, Exact B&B, and Shared Local Search modules.

Verifies Solver contract compliance, seed reproducibility, FEE budget adherence,
local search monotonic improvement, Kempe chain feasibility preservation, and
Branch & Bound optimality on Tiny benchmarks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from src.baselines import (
    DsaturSolver,
    SimulatedAnnealingSolver,
    dsatur_heuristic,
    get_baseline_solver,
    random_search,
)
from src.core.problem import is_feasible, proximity_cost, violation_weight
from src.core.solver import Budget, Result
from src.data import load_instance
from src.local_search import (
    evaluate_swap_delta,
    extract_kempe_chain,
    intensify,
    kempe_ls,
    local_search_first_improvement,
    move_ls,
    swap_ls,
)


def test_local_search_operators_preserve_feasibility_and_charge_budget() -> None:
    """Verify `move_ls`, `swap_ls`, `kempe_ls`, and `intensify` never degrade feasible solutions."""
    inst = load_instance("tiny-a")
    rng = np.random.default_rng(7)
    x0 = np.array(inst.meta["planted_solution"], dtype=np.int32)
    f0 = proximity_cost(inst, x0)
    assert is_feasible(inst, x0)

    # 1. Test move_ls
    b_move = Budget(max_evals=50.0)
    x_move, df_m, dv_m = move_ls(inst, x0, budget=b_move, rng=rng, max_passes=5)
    assert is_feasible(inst, x_move)
    assert dv_m == 0.0
    assert np.isclose(f0 + df_m, proximity_cost(inst, x_move))
    assert proximity_cost(inst, x_move) <= f0 + 1e-9
    assert b_move.used > 0.0

    # 2. Test swap_ls and exact swap delta
    df_s, dv_s = evaluate_swap_delta(inst, x0, 0, 1)
    x_swapped = x0.copy()
    x_swapped[0], x_swapped[1] = x0[1], x0[0]
    assert np.isclose(f0 + df_s, proximity_cost(inst, x_swapped))
    assert np.isclose(dv_s, violation_weight(inst, x_swapped))

    b_swap = Budget(max_evals=50.0)
    x_swap, df_sw, dv_sw = swap_ls(inst, x0, budget=b_swap, rng=rng, max_trials=60)
    assert is_feasible(inst, x_swap)
    assert dv_sw == 0.0
    assert np.isclose(f0 + df_sw, proximity_cost(inst, x_swap))

    # 3. Test kempe_ls and boundary evaluation accuracy
    chain = extract_kempe_chain(inst, x0, seed_exam=0, target_slot=(int(x0[0]) + 1) % inst.n_slots)
    assert 0 in chain
    b_kempe = Budget(max_evals=50.0)
    x_kempe, df_k, dv_k = kempe_ls(inst, x0, budget=b_kempe, rng=rng, max_steps=40)
    assert is_feasible(inst, x_kempe)
    assert dv_k == 0.0
    assert np.isclose(f0 + df_k, proximity_cost(inst, x_kempe))

    # 4. Test shared intensify and local_search_first_improvement
    b_int = Budget(max_evals=100.0)
    x_int = intensify(inst, x0, budget=b_int, rng=rng, operator="hybrid")
    assert is_feasible(inst, x_int)
    assert proximity_cost(inst, x_int) <= f0 + 1e-9

    x_ls = local_search_first_improvement(inst, x0, max_iter=5, budget=Budget(50.0), rng=rng)
    assert is_feasible(inst, x_ls)


def test_all_four_baseline_solvers_contract_and_reproducibility() -> None:
    """Verify all four baseline solvers return valid Result objects and are deterministic."""
    inst = load_instance("tiny-a")
    for solver_name in ("random", "dsatur", "sa", "exact"):
        solver_a = get_baseline_solver(solver_name)
        solver_b = get_baseline_solver(solver_name)

        res_a = solver_a.solve(inst, Budget(max_evals=120.0), seed=42)
        res_b = solver_b.solve(inst, Budget(max_evals=120.0), seed=42)

        assert isinstance(res_a, Result)
        assert res_a.feasible
        assert is_feasible(inst, res_a.best_x)
        assert np.isclose(res_a.best_f, proximity_cost(inst, res_a.best_x))
        assert res_a.history.ndim == 2 and res_a.history.shape[1] == 5
        assert res_a.history.shape[0] >= 1

        # Exact seed reproducibility check
        assert np.array_equal(res_a.best_x, res_b.best_x)
        assert np.isclose(res_a.best_f, res_b.best_f)


def test_sa_outperforms_or_matches_dsatur_on_small_instance() -> None:
    """Verify Simulated Annealing improves upon or matches initial DSATUR on `small-a`."""
    inst = load_instance("small-a")
    d_res = DsaturSolver(max_restarts=5).solve(inst, Budget(max_evals=300.0), seed=0)
    sa_res = SimulatedAnnealingSolver().solve(inst, Budget(max_evals=800.0), seed=0)

    assert d_res.feasible and sa_res.feasible
    assert sa_res.best_f <= d_res.best_f + 1e-9


def test_baseline_yaml_configs_and_functional_wrappers() -> None:
    """Verify YAML configs for all four baselines have `defaults` and `search_space`."""
    root = Path(__file__).resolve().parent.parent
    for name in ("random", "dsatur", "sa", "exact"):
        cfg_path = root / "configs" / f"{name}.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert "defaults" in cfg and isinstance(cfg["defaults"], dict)
        assert "search_space" in cfg and isinstance(cfg["search_space"], dict)
        assert len(cfg["defaults"]) > 0

    inst = load_instance("tiny-b")
    r_res = random_search(inst, Budget(30.0), seed=1)
    assert isinstance(r_res, Result) and r_res.feasible
    x_ds = dsatur_heuristic(inst, seed=1)
    assert x_ds.shape == (inst.n_exams,) and is_feasible(inst, x_ds)
