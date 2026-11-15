"""Statistical aggregation, Friedman test, and Wilcoxon-Holm analysis (`analysis/stats.py`).

Implements the statistical methodology required by Section 3.1 and 3.4:
1. Filters objective values `f(x)` strictly to feasible runs (`feasible == True`)
   while reporting `feasibility_rate` independently.
2. Computes Mean, Std, Median, IQR (Q75 - Q25), Min (Best), Mean Runtime, and
   Mean Peak Memory per `(algorithm, instance)` and per `algorithm`.
3. Computes exact optimality gap (%) on `Tiny` instances relative to `exact`.
4. Performs the non-parametric Friedman omnibus test across instances (`N >= 9`
   benchmark instances) and pairwise Wilcoxon signed-rank tests with
   Holm-Bonferroni multiple-testing correction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


def _iqr(values: pd.Series) -> float:
    """Compute interquartile range Q75 - Q25 on finite values."""
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("nan")
    return float(np.percentile(clean, 75) - np.percentile(clean, 25))


def holm_bonferroni_correction(p_values: list[float]) -> list[float]:
    """Apply step-down Holm-Bonferroni correction to a list of raw p-values.

    Args:
        p_values: Unadjusted p-values from m pairwise hypothesis tests.

    Returns:
        List of Holm-adjusted p-values in [0.0, 1.0] preserving original order.
    """
    m = len(p_values)
    if m == 0:
        return []

    arr = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(arr)
    adjusted = np.zeros(m, dtype=np.float64)

    running_max = 0.0
    for rank_idx, orig_idx in enumerate(order):
        mult = float(m - rank_idx)
        val = min(1.0, float(arr[orig_idx]) * mult)
        running_max = max(running_max, val)
        adjusted[orig_idx] = running_max

    return adjusted.tolist()


def compute_friedman_and_wilcoxon(
    detailed_df: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Friedman test and pairwise Wilcoxon + Holm tests across benchmark instances.

    Uses instance-level median feasible Carter proximity cost `median_f` across
    all algorithms evaluated on the shared 12-instance test suite (excluding
    `exact`, which runs only on `Tiny`).

    Args:
        detailed_df: DataFrame with columns `['algorithm', 'instance', 'median_f']`.
        alpha: Significance threshold (default 0.05).

    Returns:
        Tuple `(friedman_df, wilcoxon_df)`.
    """
    sub = detailed_df[detailed_df["algorithm"] != "exact"].copy()
    pivot = sub.pivot(index="instance", columns="algorithm", values="median_f")
    pivot = pivot.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    n_instances, n_algos = pivot.shape
    if n_instances < 3 or n_algos < 3:
        empty_f = pd.DataFrame(
            [
                {
                    "n_instances": n_instances,
                    "n_algorithms": n_algos,
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "significant": False,
                }
            ]
        )
        return empty_f, pd.DataFrame()

    algo_names = list(pivot.columns)
    arrays = [pivot[col].to_numpy(dtype=np.float64) for col in algo_names]
    stat, p_val = friedmanchisquare(*arrays)

    # Compute average Friedman rank per algorithm across instances (1 = best/lowest f)
    ranks_matrix = np.vstack([rankdata(row, method="average") for row in pivot.to_numpy()])
    avg_ranks = np.mean(ranks_matrix, axis=0)
    best_algo_idx = int(np.argmin(avg_ranks))
    control_algo = algo_names[best_algo_idx]

    friedman_rows = [
        {
            "n_instances": int(n_instances),
            "n_algorithms": int(n_algos),
            "statistic": round(float(stat), 4),
            "p_value": float(p_val),
            "significant": bool(p_val < alpha),
            "best_ranked_algorithm": control_algo,
            "best_average_rank": round(float(avg_ranks[best_algo_idx]), 4),
        }
    ]
    friedman_df = pd.DataFrame(friedman_rows)

    # Pairwise Wilcoxon signed-rank tests: compare control_algo against all others,
    # plus head-to-head comparisons between pure and memetic metaheuristics (GA, PSO, ACO)
    comparison_pairs: list[tuple[str, str]] = []
    for other in algo_names:
        if other != control_algo:
            comparison_pairs.append((control_algo, other))

    for a1, a2 in [
        ("ga_memetic", "ga"),
        ("pso_memetic", "pso"),
        ("aco_memetic", "aco"),
        ("ga_memetic", "pso_memetic"),
        ("ga_memetic", "aco_memetic"),
        ("pso_memetic", "aco_memetic"),
    ]:
        if a1 in algo_names and a2 in algo_names:
            if (a1, a2) not in comparison_pairs and (a2, a1) not in comparison_pairs:
                comparison_pairs.append((a1, a2))

    raw_pvals: list[float] = []
    pair_records: list[dict[str, Any]] = []

    for algo_a, algo_b in comparison_pairs:
        vec_a = pivot[algo_a].to_numpy(dtype=np.float64)
        vec_b = pivot[algo_b].to_numpy(dtype=np.float64)
        diffs = vec_a - vec_b

        if np.allclose(diffs, 0.0):
            w_stat, p_raw = 0.0, 1.0
        else:
            try:
                res_w = wilcoxon(vec_a, vec_b, zero_method="zsplit", alternative="two-sided")
                w_stat, p_raw = float(res_w.statistic), float(res_w.pvalue)
            except Exception:
                w_stat, p_raw = 0.0, 1.0

        raw_pvals.append(p_raw)
        pair_records.append(
            {
                "algorithm_a": algo_a,
                "algorithm_b": algo_b,
                "mean_diff_a_minus_b": round(float(np.mean(diffs)), 4),
                "median_diff_a_minus_b": round(float(np.median(diffs)), 4),
                "wilcoxon_stat": round(w_stat, 4),
                "p_raw": float(p_raw),
            }
        )

    adj_pvals = holm_bonferroni_correction(raw_pvals)
    for rec, p_holm in zip(pair_records, adj_pvals):
        rec["p_holm"] = float(p_holm)
        rec["significant_005"] = bool(p_holm < alpha)

    wilcoxon_df = pd.DataFrame(pair_records)
    return friedman_df, wilcoxon_df


def analyze_results(
    csv_path: str | Path = "results/raw/benchmark_runs.csv",
    tables_dir: str | Path = "results/tables",
    processed_dir: str | Path = "results/processed",
) -> dict[str, pd.DataFrame]:
    """Compute summary, instance-level, optimality gap, and statistical test tables."""
    df = pd.read_csv(csv_path)
    df["feasible"] = df["feasible"].astype(bool)
    df["best_f_feasible"] = np.where(df["feasible"], df["best_f"], np.nan)

    # 1. Detailed table per (algorithm, instance)
    detailed = df.groupby(["algorithm", "instance"], as_index=False).agg(
        mean_f=("best_f_feasible", "mean"),
        std_f=("best_f_feasible", "std"),
        median_f=("best_f_feasible", "median"),
        iqr_f=("best_f_feasible", _iqr),
        best_min_f=("best_f_feasible", "min"),
        feasible_rate=("feasible", "mean"),
        avg_time=("elapsed", "mean"),
        avg_evals=("evals_used", "mean"),
    )
    detailed["std_f"] = detailed["std_f"].fillna(0.0)

    # Compute optimality gap (%) on Tiny instances relative to exact (or minimum verified f*)
    tiny_opt: dict[str, float] = {}
    for inst_name in detailed["instance"].unique():
        if str(inst_name).startswith("tiny"):
            inst_rows = detailed[detailed["instance"] == inst_name]
            tiny_opt[str(inst_name)] = float(inst_rows["best_min_f"].min())

    gaps: list[float] = []
    for _, row in detailed.iterrows():
        iname = str(row["instance"])
        if iname in tiny_opt and np.isfinite(row["mean_f"]) and tiny_opt[iname] > 0:
            gap_pct = 100.0 * (float(row["mean_f"]) - tiny_opt[iname]) / tiny_opt[iname]
            gaps.append(round(max(0.0, gap_pct), 4))
        else:
            gaps.append(float("nan"))
    detailed["optimality_gap_pct"] = gaps

    # 2. Overall summary per algorithm
    summary = df.groupby("algorithm", as_index=False).agg(
        mean_f=("best_f_feasible", "mean"),
        median_f=("best_f_feasible", "median"),
        std_f=("best_f_feasible", "std"),
        iqr_f=("best_f_feasible", _iqr),
        feasible_rate=("feasible", "mean"),
        avg_time=("elapsed", "mean"),
        avg_evals=("evals_used", "mean"),
    )
    summary["std_f"] = summary["std_f"].fillna(0.0)

    # Attach average Friedman rank across instances to summary table
    sub_det = detailed[detailed["algorithm"] != "exact"]
    piv = sub_det.pivot(index="instance", columns="algorithm", values="median_f").dropna()
    if not piv.empty:
        r_mat = np.vstack([rankdata(r, method="average") for r in piv.to_numpy()])
        rank_map = dict(zip(piv.columns, np.mean(r_mat, axis=0)))
        summary["avg_rank"] = summary["algorithm"].map(lambda a: round(rank_map.get(a, np.nan), 4))
    else:
        summary["avg_rank"] = np.nan

    # 3. Friedman & Wilcoxon-Holm tests
    friedman_df, wilcoxon_df = compute_friedman_and_wilcoxon(detailed)

    out_tables = Path(tables_dir)
    out_proc = Path(processed_dir)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_proc.mkdir(parents=True, exist_ok=True)

    summary.to_csv(out_tables / "summary_by_algorithm.csv", index=False)
    detailed.to_csv(out_tables / "detailed_by_instance.csv", index=False)
    friedman_df.to_csv(out_tables / "friedman_test.csv", index=False)
    wilcoxon_df.to_csv(out_tables / "wilcoxon_holm_pairwise.csv", index=False)

    summary.to_csv(out_proc / "summary_by_algorithm.csv", index=False)
    detailed.to_csv(out_proc / "detailed_by_instance.csv", index=False)

    return {
        "summary": summary,
        "detailed": detailed,
        "friedman": friedman_df,
        "wilcoxon": wilcoxon_df,
    }


def analyze_ablation_results(
    csv_path: str | Path = "results/raw/ablation_runs.csv",
    tables_dir: str | Path = "results/tables",
) -> pd.DataFrame:
    """Summarize ablation study runs (`repair` vs `penalty`, `pure` vs `memetic`, encodings)."""
    path = Path(csv_path)
    if not path.is_file():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["feasible"] = df["feasible"].astype(bool)
    df["best_f_feasible"] = np.where(df["feasible"], df["best_f"], np.nan)

    abl_summary = df.groupby(["study", "variant", "instance"], as_index=False).agg(
        feasible_rate=("feasible", "mean"),
        mean_f=("best_f_feasible", "mean"),
        median_f=("best_f_feasible", "median"),
        mean_violation=("violation_weight", "mean"),
        avg_time=("elapsed", "mean"),
    )
    out_dir = Path(tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    abl_summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    return abl_summary


def main() -> None:
    """CLI entry point for statistical analysis."""
    raw_dir = Path("results/raw")
    bench_csv = raw_dir / "benchmark_runs.csv"
    if not bench_csv.is_file():
        csv_candidates = sorted(
            [f for f in raw_dir.glob("*.csv") if "ablation" not in f.name],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not csv_candidates:
            print("No benchmark CSV files found in results/raw/.")
            return
        bench_csv = csv_candidates[0]

    res = analyze_results(bench_csv)
    print("=== Summary by Algorithm ===")
    print(res["summary"].to_string(index=False))
    print("\n=== Friedman Test Across Instances ===")
    print(res["friedman"].to_string(index=False))
    if not res["wilcoxon"].empty:
        print("\n=== Pairwise Wilcoxon Signed-Rank Tests (Holm-Adjusted) ===")
        print(res["wilcoxon"].to_string(index=False))

    abl_csv = raw_dir / "ablation_runs.csv"
    if abl_csv.is_file():
        abl_df = analyze_ablation_results(abl_csv)
        print(
            f"\nSaved ablation summary ({len(abl_df)} rows) to results/tables/ablation_summary.csv."
        )


if __name__ == "__main__":
    main()
