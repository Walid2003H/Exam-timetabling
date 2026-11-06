"""Unit tests for P8 Experiments, Hyperparameter Tuning, Ablations, and Statistical Analysis.

Verifies solver factory registry (`get_solver`), candidate hyperparameter sampling,
Holm-Bonferroni step-down p-value adjustment, Friedman and Wilcoxon statistical
pipeline, and presence/schema of `experiments/config.yaml` and `experiments/ablations.yaml`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from analysis.stats import (
    analyze_ablation_results,
    analyze_results,
    compute_friedman_and_wilcoxon,
    holm_bonferroni_correction,
)
from experiments.run import SOLVER_REGISTRY, get_solver, load_instance_from_npz
from experiments.tune import sample_candidate_configs, tune_solver
from src.core.solver import Budget


def test_solver_registry_instantiates_all_ten_canonical_solvers() -> None:
    """Verify all 10 canonical solvers instantiate via `get_solver` and run on `tiny-a`."""
    expected_names = {
        "random",
        "dsatur",
        "sa",
        "exact",
        "ga",
        "ga_memetic",
        "pso",
        "pso_memetic",
        "aco",
        "aco_memetic",
    }
    assert set(SOLVER_REGISTRY.keys()) == expected_names

    inst = load_instance_from_npz("data/generated/tiny-a.npz")
    for name in sorted(expected_names):
        solver = get_solver(name, use_tuned=True)
        res = solver.solve(inst, Budget(max_evals=40.0), seed=0)
        assert res.feasible
        assert np.isfinite(res.best_f)

    with pytest.raises(ValueError, match="Unknown algorithm"):
        get_solver("non_existent_solver")


def test_hyperparameter_candidate_sampling_and_single_solver_tuning(tmp_path: Path) -> None:
    """Verify `sample_candidate_configs` and `tune_solver` produce valid tuned YAML files."""
    rng = np.random.default_rng(1000)
    defaults = {"pop_size": 24, "mutation_rate": 0.04}
    space = {"pop_size": [16, 24, 32], "mutation_rate": [0.02, 0.04]}

    cands = sample_candidate_configs(defaults, space, max_candidates=4, rng=rng)
    assert len(cands) == 4
    assert cands[0] == defaults

    best_hp, recs = tune_solver(
        "random",
        configs_dir="configs",
        tuned_dir=tmp_path,
        generated_dir="data/generated",
        tuning_seeds=[1000, 1001],
        max_candidates=2,
        tuning_fee=30.0,
    )
    assert isinstance(best_hp, dict)
    assert len(recs) == 2
    assert (tmp_path / "random.yaml").is_file()


def test_holm_bonferroni_and_friedman_wilcoxon_pipeline(tmp_path: Path) -> None:
    """Verify Holm-Bonferroni correction and Friedman + Wilcoxon tests on benchmark data."""
    raw_p = [0.01, 0.04, 0.03, 0.20]
    adj_p = holm_bonferroni_correction(raw_p)
    assert len(adj_p) == 4
    # Smallest p=0.01 multiplied by 4 -> 0.04; second smallest p=0.03 * 3 -> 0.09
    assert np.isclose(adj_p[0], 0.04)
    assert np.isclose(adj_p[2], 0.09)
    assert all(0.0 <= p <= 1.0 for p in adj_p)

    # Test on synthetic 10-instance, 4-algorithm table
    rows = []
    for i in range(10):
        rows.append({"algorithm": "pso_memetic", "instance": f"inst-{i}", "median_f": 20.0 + i})
        rows.append({"algorithm": "aco_memetic", "instance": f"inst-{i}", "median_f": 20.8 + i})
        rows.append({"algorithm": "ga_memetic", "instance": f"inst-{i}", "median_f": 21.5 + i})
        rows.append({"algorithm": "random", "instance": f"inst-{i}", "median_f": 24.0 + i})

    det_df = pd.DataFrame(rows)
    f_df, w_df = compute_friedman_and_wilcoxon(det_df)
    assert f_df["n_instances"].iloc[0] == 10
    assert f_df["significant"].iloc[0]
    assert f_df["best_ranked_algorithm"].iloc[0] == "pso_memetic"
    assert not w_df.empty
    assert "p_holm" in w_df.columns


def test_experiments_configs_and_saved_results_integrity() -> None:
    """Verify `experiments/config.yaml`, `ablations.yaml`, and generated tables are valid."""
    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "experiments" / "config.yaml").read_text(encoding="utf-8"))
    abl = yaml.safe_load((root / "experiments" / "ablations.yaml").read_text(encoding="utf-8"))

    assert len(cfg["seeds"]) == 30
    assert len(cfg["algorithms"]) == 10
    assert len(cfg["instances"]) == 12
    assert len(abl["studies"]) >= 3

    if (root / "results" / "raw" / "benchmark_runs.csv").is_file():
        res = analyze_results(
            root / "results" / "raw" / "benchmark_runs.csv",
            tables_dir=root / "results" / "tables",
            processed_dir=root / "results" / "processed",
        )
        assert len(res["summary"]) == 10
        assert (res["summary"]["feasible_rate"] == 1.0).all()

    if (root / "results" / "raw" / "ablation_runs.csv").is_file():
        abl_df = analyze_ablation_results(
            root / "results" / "raw" / "ablation_runs.csv",
            tables_dir=root / "results" / "tables",
        )
        assert not abl_df.empty
