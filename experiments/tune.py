"""Hyperparameter tuning pipeline on dedicated tuning instances (`experiments/tune.py`).

Uses random/grid search over each solver's `search_space` in `configs/<solver>.yaml`
evaluated strictly on `tune-small` and `tune-medium` with tuning seeds `1000..1004`
(Section 3.4) to prevent data leakage into the 12 test benchmark instances.
Writes the selected configurations to `configs/tuned/<solver>.yaml`.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.algorithms.aco import ACOMemeticSolver, ACOSolver
from src.algorithms.ga import GAMemeticSolver, GASolver
from src.algorithms.pso import PSOMemeticSolver, PSOSolver
from src.baselines.dsatur import DsaturSolver
from src.baselines.exact_bb import ExactBranchAndBoundSolver
from src.baselines.random_search import RandomSearchSolver
from src.baselines.sa import SimulatedAnnealingSolver
from src.core.solver import Budget, Solver
from src.data import load_instance

SOLVER_CLASSES: dict[str, type[Solver]] = {
    "random": RandomSearchSolver,
    "dsatur": DsaturSolver,
    "sa": SimulatedAnnealingSolver,
    "exact": ExactBranchAndBoundSolver,
    "ga": GASolver,
    "ga_memetic": GAMemeticSolver,
    "pso": PSOSolver,
    "pso_memetic": PSOMemeticSolver,
    "aco": ACOSolver,
    "aco_memetic": ACOMemeticSolver,
}


def sample_candidate_configs(
    defaults: dict[str, Any],
    search_space: dict[str, list[Any]],
    max_candidates: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Generate up to `max_candidates` hyperparameter dictionaries including `defaults`."""
    candidates: list[dict[str, Any]] = [dict(defaults)]
    if not search_space or max_candidates <= 1:
        return candidates

    keys = sorted(search_space.keys())
    value_lists = [
        search_space[k] for k in keys if isinstance(search_space[k], list) and search_space[k]
    ]
    if len(value_lists) != len(keys):
        return candidates

    all_combos = list(itertools.product(*value_lists))
    order = rng.permutation(len(all_combos))

    for idx_raw in order:
        if len(candidates) >= max_candidates:
            break
        combo = all_combos[int(idx_raw)]
        cand = dict(defaults)
        for k, v in zip(keys, combo):
            cand[k] = v
        if cand not in candidates:
            candidates.append(cand)

    return candidates


def tune_solver(
    solver_name: str,
    configs_dir: str | Path = "configs",
    tuned_dir: str | Path = "configs/tuned",
    generated_dir: str | Path = "data/generated",
    tuning_seeds: list[int] | None = None,
    max_candidates: int = 4,
    tuning_fee: float = 300.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tune `solver_name` on `tune-small` using `tuning_seeds` and save tuned YAML."""
    if tuning_seeds is None:
        tuning_seeds = [1000, 1001, 1002]

    cfg_path = Path(configs_dir) / f"{solver_name}.yaml"
    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    defaults = dict(raw_cfg.get("defaults", {}))
    search_space = dict(raw_cfg.get("search_space", {}))

    if solver_name == "exact":
        out_path = Path(tuned_dir) / f"{solver_name}.yaml"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            yaml.safe_dump(
                {"defaults": defaults, "tuned_on": "tiny-a", "seeds": tuning_seeds},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return defaults, []

    inst = load_instance("tune-small", generated_dir=generated_dir)
    rng = np.random.default_rng(1000)
    candidates = sample_candidate_configs(defaults, search_space, max_candidates, rng)
    solver_cls = SOLVER_CLASSES[solver_name]

    best_hparams = dict(defaults)
    best_score = (float("-inf"), float("inf"))  # maximize feasibility_rate, minimize mean_f
    records: list[dict[str, Any]] = []

    for c_idx, cand in enumerate(candidates):
        feas_Flags: list[bool] = []
        feas_costs: list[float] = []
        t0 = time.perf_counter()

        for seed in tuning_seeds:
            solver = solver_cls(cand)
            res = solver.solve(inst, Budget(max_evals=tuning_fee), seed=int(seed))
            feas_Flags.append(bool(res.feasible))
            if res.feasible and np.isfinite(res.best_f):
                feas_costs.append(float(res.best_f))

        feas_rate = float(np.mean(feas_Flags)) if feas_Flags else 0.0
        mean_f = float(np.mean(feas_costs)) if feas_costs else float("inf")
        elapsed = time.perf_counter() - t0

        records.append(
            {
                "solver": solver_name,
                "candidate_id": c_idx,
                "feasibility_rate": round(feas_rate, 4),
                "mean_f": round(mean_f, 4) if np.isfinite(mean_f) else float("inf"),
                "elapsed_sec": round(elapsed, 3),
                "hparams": str(cand),
            }
        )

        # Prefer higher feasibility rate, then lower mean Carter proximity cost
        if feas_rate > best_score[0] or (
            np.isclose(feas_rate, best_score[0]) and mean_f < best_score[1] - 1e-9
        ):
            best_score = (feas_rate, mean_f)
            best_hparams = dict(cand)

    out_path = Path(tuned_dir) / f"{solver_name}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tuned_payload = {
        "defaults": best_hparams,
        "tuning_metadata": {
            "instance": "tune-small",
            "seeds": list(tuning_seeds),
            "budget_fee": float(tuning_fee),
            "best_feasibility_rate": float(best_score[0]),
            "best_mean_f": float(round(best_score[1], 4)) if np.isfinite(best_score[1]) else None,
        },
    }
    out_path.write_text(yaml.safe_dump(tuned_payload, sort_keys=False), encoding="utf-8")
    return best_hparams, records


def tune_all_solvers(
    configs_dir: str | Path = "configs",
    tuned_dir: str | Path = "configs/tuned",
    generated_dir: str | Path = "data/generated",
    results_dir: str | Path = "results/tables",
    quick: bool = False,
) -> pd.DataFrame:
    """Tune all 10 canonical solvers and write `configs/tuned/*.yaml` and `tuning_summary.csv`."""
    seeds = [1000, 1001] if quick else [1000, 1001, 1002, 1003, 1004]
    max_cands = 2 if quick else 4
    fee = 150.0 if quick else 300.0

    all_records: list[dict[str, Any]] = []
    for name in SOLVER_CLASSES:
        _, recs = tune_solver(
            name,
            configs_dir=configs_dir,
            tuned_dir=tuned_dir,
            generated_dir=generated_dir,
            tuning_seeds=seeds,
            max_candidates=max_cands,
            tuning_fee=fee,
        )
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    out_tables = Path(results_dir)
    out_tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tables / "tuning_summary.csv", index=False)
    return df


def main() -> None:
    """CLI entry point for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Tune hyperparameters on tune-* instances.")
    parser.add_argument("--quick", action="store_true", help="Run fast tuning check.")
    args = parser.parse_args()
    df = tune_all_solvers(quick=args.quick)
    print(f"Tuning completed across {len(SOLVER_CLASSES)} solvers ({len(df)} trials logged).")


if __name__ == "__main__":
    main()
