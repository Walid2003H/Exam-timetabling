"""Unit tests for P6 Particle Swarm Optimization (`src/algorithms/pso.py`).

Tests Random-Keys and Direct Discrete encodings, pure `PSOSolver` (`pso`) and
memetic `PSOMemeticSolver` (`pso_memetic`), repair vs penalty constraint handling,
seed reproducibility, FEE budget adherence, and YAML configuration schemas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.algorithms.pso import (
    PSOMemeticSolver,
    PSOSolver,
    decode_random_keys,
    encode_slots_to_keys,
    update_direct_discrete_particle,
)
from src.core.problem import is_feasible, proximity_cost
from src.core.solver import Budget, Result
from src.data import load_instance


def test_random_keys_and_direct_discrete_operators() -> None:
    """Verify `decode_random_keys`, `encode_slots_to_keys`, and discrete update bounds."""
    inst = load_instance("tiny-a")
    rng = np.random.default_rng(123)

    keys = rng.random(inst.n_exams)
    x_greedy = decode_random_keys(inst, keys, rng, decode_mode="priority_greedy")
    assert x_greedy.shape == (inst.n_exams,)
    assert x_greedy.dtype == np.int32
    assert np.all((x_greedy >= 0) & (x_greedy < inst.n_slots))

    x_map = decode_random_keys(inst, keys, rng, decode_mode="slot_map")
    assert x_map.shape == (inst.n_exams,)
    assert np.all((x_map >= 0) & (x_map < inst.n_slots))

    back_keys = encode_slots_to_keys(inst, x_greedy, rng, decode_mode="priority_greedy")
    assert back_keys.shape == (inst.n_exams,)
    assert np.all((back_keys >= 0.0) & (back_keys <= 1.0))

    vel_prob = np.full(inst.n_exams, 0.3, dtype=np.float64)
    pbest = rng.integers(0, inst.n_slots, size=inst.n_exams, dtype=np.int32)
    gbest = rng.integers(0, inst.n_slots, size=inst.n_exams, dtype=np.int32)
    new_x, new_vel = update_direct_discrete_particle(
        x_greedy,
        vel_prob,
        pbest,
        gbest,
        n_slots=inst.n_slots,
        w=0.7,
        c1=1.4,
        c2=1.4,
        mutation_rate=0.05,
        rng=rng,
    )
    assert new_x.shape == (inst.n_exams,)
    assert np.all((new_x >= 0) & (new_x < inst.n_slots))
    assert np.all((new_vel >= 0.05) & (new_vel <= 0.95))


def test_pso_both_encodings_and_reproducibility() -> None:
    """Verify `PSOSolver` works with both `random_keys` and `direct_discrete` encodings."""
    inst = load_instance("tiny-a")

    for enc in ("random_keys", "direct_discrete"):
        solver_1 = PSOSolver(encoding=enc, swarm_size=10)
        solver_2 = PSOSolver(encoding=enc, swarm_size=10)

        res_1 = solver_1.solve(inst, Budget(max_evals=80.0), seed=17)
        res_2 = solver_2.solve(inst, Budget(max_evals=80.0), seed=17)

        assert isinstance(res_1, Result)
        assert res_1.feasible
        assert is_feasible(inst, res_1.best_x)
        assert np.isclose(res_1.best_f, proximity_cost(inst, res_1.best_x))
        assert res_1.history.ndim == 2 and res_1.history.shape[1] == 5
        assert res_1.history.shape[0] >= 1

        # Deterministic reproducibility with identical seed
        assert np.array_equal(res_1.best_x, res_2.best_x)
        assert np.isclose(res_1.best_f, res_2.best_f)


def test_pso_memetic_improves_over_or_matches_pure_pso() -> None:
    """Verify `PSOMemeticSolver` (`pso_memetic`) finds strong feasible schedules on `small-a`."""
    inst = load_instance("small-a")
    pure_res = PSOSolver(swarm_size=12).solve(inst, Budget(max_evals=400.0), seed=0)
    mem_res = PSOMemeticSolver(swarm_size=12).solve(inst, Budget(max_evals=400.0), seed=0)

    assert pure_res.feasible and mem_res.feasible
    assert is_feasible(inst, mem_res.best_x)
    assert mem_res.best_f <= pure_res.best_f + 1e-9


def test_pso_repair_vs_penalty_ablation_modes() -> None:
    """Verify both `constraint_handling='repair'` and `'penalty'` execute cleanly."""
    inst = load_instance("tiny-b")
    res_rep = PSOSolver(constraint_handling="repair", swarm_size=10).solve(
        inst, Budget(max_evals=60.0), seed=5
    )
    res_pen = PSOSolver(constraint_handling="penalty", decode_mode="slot_map", swarm_size=10).solve(
        inst, Budget(max_evals=60.0), seed=5
    )

    assert res_rep.feasible
    assert isinstance(res_pen, Result)
    if not res_pen.feasible:
        assert res_pen.best_f == float("inf")

    with pytest.raises(ValueError, match="Unknown PSO encoding"):
        PSOSolver(encoding="invalid_encoding").solve(inst, Budget(max_evals=10.0), seed=0)


def test_pso_yaml_configs_schema() -> None:
    """Verify pso.yaml and pso_memetic.yaml contain defaults and search_space."""
    root = Path(__file__).resolve().parent.parent
    for fname in ("pso.yaml", "pso_memetic.yaml"):
        cfg = yaml.safe_load((root / "configs" / fname).read_text(encoding="utf-8"))
        assert "defaults" in cfg and isinstance(cfg["defaults"], dict)
        assert "search_space" in cfg and isinstance(cfg["search_space"], dict)
        assert "swarm_size" in cfg["defaults"]
        assert "encoding" in cfg["search_space"]
