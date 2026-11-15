"""Unified experiment and ablation runner (`experiments/run.py`).

Executes all 10 canonical solvers across the benchmark suite (`tiny`, `small`,
`medium`, `large`) using deterministic seeds (`0..29` in full mode, `0..4` in
pilot mode, or `0..1` in `--quick` mode), logs wall-clock runtime and peak memory
allocation (`tracemalloc`), verifies solution feasibility, and saves:
- `results/raw/benchmark_runs.csv` (and timestamped archive copy)
- `results/raw/ablation_runs.csv` (for the P5/P6/P7/P8 ablation studies)
- `results/processed/convergence_trajectories.npz` (interpolated 201-point FEE
  convergence & diversity curves for P9 figure generation).
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.algorithms.aco import ACOMemeticSolver, ACOSolver
from src.algorithms.ga import GAMemeticSolver, GASolver
from src.algorithms.pso import PSOMemeticSolver, PSOSolver
from src.baselines.dsatur import DsaturSolver
from src.baselines.exact_bb import ExactBranchAndBoundSolver
from src.baselines.random_search import RandomSearchSolver
from src.baselines.sa import SimulatedAnnealingSolver
from src.core.instance import Instance
from src.core.problem import proximity_cost, violation_weight
from src.core.solver import Budget, Solver
from src.data import load_instance_npz

SOLVER_REGISTRY: dict[str, type[Solver]] = {
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


def load_instance_from_npz(path: str | Path) -> Instance:
    """Load an `Instance` from a compressed `.npz` benchmark file."""
    return load_instance_npz(path)


def load_solver_hparams(
    algo_name: str,
    use_tuned: bool = True,
    configs_dir: str | Path = "configs",
) -> dict[str, Any]:
    """Load default or tuned hyperparameters for `algo_name` from `configs/`."""
    base_dir = Path(configs_dir)
    tuned_path = base_dir / "tuned" / f"{algo_name}.yaml"
    default_path = base_dir / f"{algo_name}.yaml"

    target = tuned_path if (use_tuned and tuned_path.is_file()) else default_path
    if not target.is_file():
        return {}
    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return dict(data.get("defaults", {}))


def get_solver(
    algo_name: str,
    use_tuned: bool = True,
    configs_dir: str | Path = "configs",
    overrides: dict[str, Any] | None = None,
) -> Solver:
    """Instantiate a registered solver by its canonical name with YAML hyperparameters."""
    key = algo_name.strip().lower()
    if key not in SOLVER_REGISTRY:
        raise ValueError(
            f"Unknown algorithm {algo_name!r}; expected one of {sorted(SOLVER_REGISTRY)}."
        )
    hparams = load_solver_hparams(key, use_tuned=use_tuned, configs_dir=configs_dir)
    if overrides:
        hparams.update(overrides)
    return SOLVER_REGISTRY[key](hparams)


def _interpolate_trajectory(
    history: np.ndarray,
    max_evals: float,
    n_points: int = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate `history` onto a uniform FEE grid `[0, max_evals]` of length `n_points`."""
    grid = np.linspace(0.0, float(max_evals), n_points, dtype=np.float64)
    if history.ndim != 2 or history.shape[0] == 0:
        return grid, np.full(n_points, np.nan), np.zeros(n_points)

    evals = history[:, 0]
    best_f = np.where(np.isfinite(history[:, 2]), history[:, 2], np.nan)
    div = history[:, 4]

    interp_f = np.full(n_points, np.nan, dtype=np.float64)
    interp_div = np.zeros(n_points, dtype=np.float64)

    for i, g_val in enumerate(grid):
        idx = int(np.searchsorted(evals, g_val, side="right")) - 1
        idx = max(0, min(history.shape[0] - 1, idx))
        interp_f[i] = best_f[idx]
        interp_div[i] = div[idx]

    return grid, interp_f, interp_div


def run_benchmark_suite(
    config_path: str | Path = "experiments/config.yaml",
    mode: str = "pilot",
    max_seeds: int | None = None,
    output_dir: str | Path = "results",
) -> list[dict[str, Any]]:
    """Run the benchmark experiment matrix and save raw CSVs and convergence curves.

    Args:
        config_path: Path to `experiments/config.yaml`.
        mode: One of `"full"`, `"pilot"`, or `"quick"`.
        max_seeds: Optional cap on the number of seeds executed per pair.
        output_dir: Root results directory (`results/`).

    Returns:
        List of run record dictionaries written to `results/raw/benchmark_runs.csv`.
    """
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if mode == "quick":
        seeds = list(cfg.get("quick_seeds", [0, 1]))
        fee_key = "quick_fee"
    elif mode == "pilot":
        seeds = list(cfg.get("pilot_seeds", [0, 1, 2, 3, 4]))
        fee_key = "pilot_fee"
    else:
        seeds = list(cfg.get("seeds", list(range(30))))
        fee_key = "full_fee"

    if max_seeds is not None:
        seeds = seeds[: int(max_seeds)]

    use_tuned = bool(cfg.get("use_tuned_configs", True))
    budgets_by_group = cfg.get("budgets_by_group", {})
    algorithms = list(cfg.get("algorithms", []))
    instances_cfg = list(cfg.get("instances", []))

    rows: list[dict[str, Any]] = []
    traj_store: dict[str, np.ndarray] = {}

    for inst_item in instances_cfg:
        inst_name = str(inst_item["name"])
        inst_group = str(inst_item.get("group", inst_name.split("-")[0]))
        inst_path = Path(inst_item["path"])
        if not inst_path.is_file():
            continue

        inst = load_instance_from_npz(inst_path)
        grp_budget_cfg = budgets_by_group.get(inst_group, {})
        max_evals = float(
            grp_budget_cfg.get(
                fee_key,
                cfg.get("budget", {}).get("max_evals", 1000.0),
            )
        )

        for algo_name in algorithms:
            # Exact Branch & Bound is evaluated on Tiny instances (Section 1.1 & 3.4)
            if algo_name == "exact" and inst_group != "tiny":
                continue

            f_curves: list[np.ndarray] = []
            div_curves: list[np.ndarray] = []
            grid_ref: np.ndarray | None = None

            for seed in seeds:
                solver = get_solver(algo_name, use_tuned=use_tuned)
                budget = Budget(max_evals=max_evals)

                result = solver.solve(inst, budget, int(seed))
                peak_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0

                # Independent contract verification of feasibility and objective f(x)
                v_actual = float(violation_weight(inst, result.best_x))
                is_feas = bool(result.feasible) and (v_actual <= 1e-9)
                f_actual = float(proximity_cost(inst, result.best_x)) if is_feas else float("inf")

                grid, f_interp, div_interp = _interpolate_trajectory(
                    result.history, max_evals=max_evals
                )
                grid_ref = grid
                f_curves.append(f_interp)
                div_curves.append(div_interp)

                rows.append(
                    {
                        "algorithm": algo_name,
                        "instance": inst_name,
                        "group": inst_group,
                        "seed": int(seed),
                        "best_f": round(f_actual, 6) if is_feas else float("inf"),
                        "violation_weight": round(v_actual, 4),
                        "feasible": is_feas,
                        "evals_used": round(float(result.evals), 4),
                        "budget_fee": float(max_evals),
                        "elapsed": round(float(result.elapsed), 6),
                        "peak_memory_mb": round(peak_mb, 4),
                        "hparams": json.dumps(result.hparams, sort_keys=True),
                    }
                )

            if grid_ref is not None and f_curves:
                key_prefix = f"{inst_name}__{algo_name}"
                traj_store[f"{key_prefix}__grid"] = grid_ref
                traj_store[f"{key_prefix}__f"] = np.asarray(f_curves, dtype=np.float64)
                traj_store[f"{key_prefix}__div"] = np.asarray(div_curves, dtype=np.float64)

    out_root = Path(output_dir)
    raw_dir = out_root / "raw"
    proc_dir = out_root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    canonical_csv = raw_dir / "benchmark_runs.csv"
    ts_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{mode}_{len(seeds)}seeds.csv"
    archive_csv = raw_dir / ts_name

    if rows:
        fieldnames = list(rows[0].keys())
        for target_csv in (canonical_csv, archive_csv):
            with target_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    if traj_store:
        np.savez_compressed(proc_dir / "convergence_trajectories.npz", **traj_store)

    return rows


def run_ablation_suite(
    ablation_path: str | Path = "experiments/ablations.yaml",
    quick: bool = False,
    output_dir: str | Path = "results",
) -> list[dict[str, Any]]:
    """Run all ablation studies defined in `experiments/ablations.yaml`."""
    cfg = yaml.safe_load(Path(ablation_path).read_text(encoding="utf-8"))
    seeds = list(cfg.get("quick_seeds", [0, 1]) if quick else cfg.get("seeds", [0, 1, 2, 3, 4]))
    instances_cfg = list(cfg.get("instances", []))
    studies = list(cfg.get("studies", []))

    rows: list[dict[str, Any]] = []
    for study in studies:
        study_name = str(study["study_name"])
        for variant in study.get("variants", []):
            label = str(variant["label"])
            solver_name = str(variant["solver"])
            overrides = dict(variant.get("hparams", {}))

            for inst_item in instances_cfg:
                inst_name = str(inst_item["name"])
                inst_group = str(inst_item.get("group", inst_name.split("-")[0]))
                inst_path = Path(inst_item["path"])
                if not inst_path.is_file():
                    continue
                inst = load_instance_from_npz(inst_path)
                fee = float(inst_item.get("fee", 500.0))
                if quick:
                    fee = min(fee, 200.0)

                for seed in seeds:
                    solver = get_solver(solver_name, use_tuned=True, overrides=overrides)
                    res = solver.solve(inst, Budget(max_evals=fee), int(seed))
                    v_val = float(violation_weight(inst, res.best_x))
                    is_feas = bool(res.feasible) and (v_val <= 1e-9)
                    f_val = float(proximity_cost(inst, res.best_x)) if is_feas else float("inf")

                    rows.append(
                        {
                            "study": study_name,
                            "variant": label,
                            "solver": solver_name,
                            "instance": inst_name,
                            "group": inst_group,
                            "seed": int(seed),
                            "best_f": round(f_val, 6) if is_feas else float("inf"),
                            "violation_weight": round(v_val, 4),
                            "feasible": is_feas,
                            "evals_used": round(float(res.evals), 4),
                            "elapsed": round(float(res.elapsed), 6),
                        }
                    )

    raw_dir = Path(output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_csv = raw_dir / "ablation_runs.csv"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return rows


def main() -> None:
    """CLI entry point for running benchmark and ablation experiments."""
    parser = argparse.ArgumentParser(description="Run Exam Timetabling benchmark experiments.")
    parser.add_argument(
        "--config",
        default="experiments/config.yaml",
        help="Path to benchmark config YAML.",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "pilot", "quick"],
        default="pilot",
        help="Execution mode (full=30 seeds, pilot=5 seeds, quick=2 seeds).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shortcut for --mode quick.",
    )
    parser.add_argument(
        "--skip-ablations",
        action="store_true",
        help="Skip running the ablation suite.",
    )
    args = parser.parse_args()
    run_mode = "quick" if args.quick else args.mode

    bench_rows = run_benchmark_suite(config_path=args.config, mode=run_mode)
    print(f"Completed {len(bench_rows)} benchmark runs (mode={run_mode}).")

    if not args.skip_ablations and Path("experiments/ablations.yaml").is_file():
        abl_rows = run_ablation_suite(quick=(run_mode == "quick"))
        print(f"Completed {len(abl_rows)} ablation runs.")


if __name__ == "__main__":
    main()
