"""Unit tests for P7 Ant Colony Optimization (`src/algorithms/aco.py`).

Tests MMAS pheromone bounds and clamping `[tau_min, tau_max]`, single-ant
construction, pure `ACOSolver` (`aco`) and memetic `ACOMemeticSolver`
(`aco_memetic`), repair vs penalty constraint handling, deterministic seed
reproducibility, and YAML configuration schemas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.algorithms.aco import (
    ACOMemeticSolver,
    ACOSolver,
    compute_mmas_bounds,
    construct_ant_timetable,
    update_mmas_pheromones,
)
from src.core.problem import is_feasible, proximity_cost
from src.core.solver import Budget, Result
from src.data import load_instance


def test_mmas_bounds_and_pheromone_clamping() -> None:
    """Verify `compute_mmas_bounds`, `update_mmas_pheromones`, and `construct_ant_timetable`."""
    inst = load_instance("tiny-a")
    rng = np.random.default_rng(99)

    tau_min, tau_max = compute_mmas_bounds(best_cost=25.0, rho=0.12, n_slots=inst.n_slots)
    assert 0.0 < tau_min < tau_max

    tau = np.full((inst.n_exams, inst.n_slots), tau_max, dtype=np.float64)
    ant_x = construct_ant_timetable(inst, tau, alpha=1.0, beta=2.5, q0=0.15, rng=rng)
    assert ant_x.shape == (inst.n_exams,)
    assert ant_x.dtype == np.int32
    assert np.all((ant_x >= 0) & (ant_x < inst.n_slots))

    # Update pheromones multiple times and verify strict clamping in [tau_min, tau_max]
    for _ in range(10):
        update_mmas_pheromones(
            tau, ant_x, elite_cost=25.0, rho=0.15, tau_min=tau_min, tau_max=tau_max
        )
    assert np.all(tau >= tau_min - 1e-12)
    assert np.all(tau <= tau_max + 1e-12)


def test_aco_and_aco_memetic_reproducibility_and_feasibility() -> None:
    """Verify `ACOSolver` and `ACOMemeticSolver` return valid Results and are deterministic."""
    inst = load_instance("tiny-a")

    for solver_cls in (ACOSolver, ACOMemeticSolver):
        s1 = solver_cls(n_ants=8)
        s2 = solver_cls(n_ants=8)

        res_1 = s1.solve(inst, Budget(max_evals=80.0), seed=21)
        res_2 = s2.solve(inst, Budget(max_evals=80.0), seed=21)

        assert isinstance(res_1, Result)
        assert res_1.feasible
        assert is_feasible(inst, res_1.best_x)
        assert np.isclose(res_1.best_f, proximity_cost(inst, res_1.best_x))
        assert res_1.history.ndim == 2 and res_1.history.shape[1] == 5
        assert res_1.history.shape[0] >= 1

        # Exact reproducibility for identical seed
        assert np.array_equal(res_1.best_x, res_2.best_x)
        assert np.isclose(res_1.best_f, res_2.best_f)


def test_aco_memetic_refines_solutions_on_small_instance() -> None:
    """Verify `ACOMemeticSolver` (`aco_memetic`) finds feasible low-cost schedules on `small-a`."""
    inst = load_instance("small-a")
    pure_res = ACOSolver(n_ants=10).solve(inst, Budget(max_evals=250.0), seed=0)
    mem_res = ACOMemeticSolver(n_ants=10).solve(inst, Budget(max_evals=250.0), seed=0)

    assert pure_res.feasible and mem_res.feasible
    assert is_feasible(inst, mem_res.best_x)
    assert np.isclose(mem_res.best_f, proximity_cost(inst, mem_res.best_x))
    assert mem_res.best_f <= pure_res.best_f + 1e-9


def test_aco_repair_vs_penalty_modes() -> None:
    """Verify both `constraint_handling='repair'` and `'penalty'` execute cleanly."""
    inst = load_instance("tiny-b")
    res_rep = ACOSolver(constraint_handling="repair", n_ants=8).solve(
        inst, Budget(max_evals=60.0), seed=3
    )
    res_pen = ACOSolver(constraint_handling="penalty", n_ants=8).solve(
        inst, Budget(max_evals=60.0), seed=3
    )

    assert res_rep.feasible
    assert isinstance(res_pen, Result)
    if not res_pen.feasible:
        assert res_pen.best_f == float("inf")

    with pytest.raises(ValueError, match="Unknown constraint_handling"):
        ACOSolver(constraint_handling="invalid_mode").solve(inst, Budget(max_evals=10.0), seed=0)


def test_aco_yaml_configs_schema() -> None:
    """Verify aco.yaml and aco_memetic.yaml contain defaults and search_space."""
    root = Path(__file__).resolve().parent.parent
    for fname in ("aco.yaml", "aco_memetic.yaml"):
        cfg = yaml.safe_load((root / "configs" / fname).read_text(encoding="utf-8"))
        assert "defaults" in cfg and isinstance(cfg["defaults"], dict)
        assert "search_space" in cfg and isinstance(cfg["search_space"], dict)
        assert "n_ants" in cfg["defaults"]
        assert "rho" in cfg["search_space"]
