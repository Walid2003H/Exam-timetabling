"""Unit tests for P5 Genetic Algorithm (`src/algorithms/ga.py`).

Tests tournament selection, uniform and two-point crossover, slot mutation,
pure `GASolver` (`ga`) and memetic `GAMemeticSolver` (`ga_memetic`), repair vs
penalty ablation modes, deterministic seed reproducibility, and YAML schemas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.algorithms.ga import (
    GAMemeticSolver,
    GASolver,
    MemeticGASolver,
    crossover_chromosomes,
    mutate_chromosome,
    tournament_select,
)
from src.core.problem import is_feasible, proximity_cost
from src.core.solver import Budget, Result
from src.data import load_instance


def test_ga_selection_crossover_and_mutation_operators() -> None:
    """Verify `tournament_select`, `crossover_chromosomes`, and `mutate_chromosome`."""
    inst = load_instance("tiny-a")
    rng = np.random.default_rng(77)

    pop_x = rng.integers(0, inst.n_slots, size=(8, inst.n_exams), dtype=np.int32)
    pop_fit = np.array([50.0, 40.0, 12.5, 30.0, 25.0, 60.0, 19.0, 45.0])

    # With tournament_size == pop_size, the global minimum (index 2) must win
    winner = tournament_select(pop_x, pop_fit, tournament_size=8, rng=rng)
    assert np.array_equal(winner, pop_x[2])

    for cx_type in ("uniform", "two_point"):
        c1, c2 = crossover_chromosomes(pop_x[0], pop_x[1], cx_type, rng)
        assert c1.shape == (inst.n_exams,) and c2.shape == (inst.n_exams,)
        assert np.all((c1 >= 0) & (c1 < inst.n_slots))
        assert np.all((c2 >= 0) & (c2 < inst.n_slots))

    mut = mutate_chromosome(c1, inst.n_slots, mutation_rate=0.25, rng=rng)
    assert mut.shape == (inst.n_exams,)
    assert np.all((mut >= 0) & (mut < inst.n_slots))


def test_ga_and_ga_memetic_reproducibility_and_feasibility() -> None:
    """Verify `GASolver` and `GAMemeticSolver` return valid Results and are deterministic."""
    inst = load_instance("tiny-a")
    assert MemeticGASolver is GAMemeticSolver

    for solver_cls in (GASolver, GAMemeticSolver):
        s1 = solver_cls(pop_size=10)
        s2 = solver_cls(pop_size=10)

        res_1 = s1.solve(inst, Budget(max_evals=80.0), seed=19)
        res_2 = s2.solve(inst, Budget(max_evals=80.0), seed=19)

        assert isinstance(res_1, Result)
        assert res_1.feasible
        assert is_feasible(inst, res_1.best_x)
        assert np.isclose(res_1.best_f, proximity_cost(inst, res_1.best_x))
        assert res_1.history.ndim == 2 and res_1.history.shape[1] == 5
        assert res_1.history.shape[0] >= 1

        # Exact reproducibility for identical seed
        assert np.array_equal(res_1.best_x, res_2.best_x)
        assert np.isclose(res_1.best_f, res_2.best_f)


def test_ga_memetic_matches_or_improves_pure_ga_on_small_instance() -> None:
    """Verify `GAMemeticSolver` (`ga_memetic`) outperforms or matches pure `ga` on `small-a`."""
    inst = load_instance("small-a")
    pure_res = GASolver(pop_size=14).solve(inst, Budget(max_evals=300.0), seed=0)
    mem_res = GAMemeticSolver(pop_size=14).solve(inst, Budget(max_evals=300.0), seed=0)

    assert pure_res.feasible and mem_res.feasible
    assert is_feasible(inst, mem_res.best_x)
    assert mem_res.best_f <= pure_res.best_f + 1e-9


def test_ga_repair_vs_penalty_ablation_modes() -> None:
    """Verify both `constraint_handling='repair'` and `'penalty'` execute cleanly."""
    inst = load_instance("tiny-b")
    res_rep = GASolver(constraint_handling="repair", pop_size=10).solve(
        inst, Budget(max_evals=60.0), seed=11
    )
    res_pen = GASolver(constraint_handling="penalty", pop_size=10).solve(
        inst, Budget(max_evals=60.0), seed=11
    )

    assert res_rep.feasible
    assert isinstance(res_pen, Result)
    if not res_pen.feasible:
        assert res_pen.best_f == float("inf")

    with pytest.raises(ValueError, match="Unknown crossover_type"):
        GASolver(crossover_type="bad_cx").solve(inst, Budget(max_evals=10.0), seed=0)

    with pytest.raises(ValueError, match="Unknown constraint_handling"):
        GASolver(constraint_handling="bad_mode").solve(inst, Budget(max_evals=10.0), seed=0)


def test_ga_yaml_configs_schema() -> None:
    """Verify ga.yaml and ga_memetic.yaml contain defaults and search_space."""
    root = Path(__file__).resolve().parent.parent
    for fname in ("ga.yaml", "ga_memetic.yaml"):
        cfg = yaml.safe_load((root / "configs" / fname).read_text(encoding="utf-8"))
        assert "defaults" in cfg and isinstance(cfg["defaults"], dict)
        assert "search_space" in cfg and isinstance(cfg["search_space"], dict)
        assert "pop_size" in cfg["defaults"]
        assert "mutation_rate" in cfg["search_space"]
