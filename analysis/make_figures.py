"""Automated generation of benchmark figures from `results/` (`analysis/make_figures.py`)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.style import SOLVER_PALETTE, apply_plot_style


def generate_plots(
    raw_dir: str | Path = "results/raw",
    figures_dir: str | Path = "results/figures",
) -> list[Path]:
    """Generate boxplots per representative instance and quality-vs-speed trade-off chart."""
    apply_plot_style()
    rdir = Path(raw_dir)
    target_csv = rdir / "benchmark_runs.csv"
    if not target_csv.is_file():
        cands = sorted(
            [f for f in rdir.glob("*.csv") if "ablation" not in f.name],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if not cands:
            return []
        target_csv = cands[0]

    df = pd.read_csv(target_csv)
    df["feasible"] = df["feasible"].astype(bool)
    feas_df = df[df["feasible"] & np.isfinite(df["best_f"])].copy()

    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for inst in sorted(feas_df["instance"].unique()):
        inst_df = feas_df[feas_df["instance"] == inst]
        algos = [a for a in SOLVER_PALETTE if a in inst_df["algorithm"].unique()]
        if not algos:
            continue

        data = [inst_df[inst_df["algorithm"] == a]["best_f"].to_numpy() for a in algos]
        fee_val = int(inst_df["budget_fee"].iloc[0]) if "budget_fee" in inst_df.columns else 0

        fig, ax = plt.subplots(figsize=(10, 5.2))
        bp = ax.boxplot(data, tick_labels=algos, patch_artist=True)
        for patch, algo in zip(bp["boxes"], algos):
            patch.set_facecolor(SOLVER_PALETTE.get(algo, "#3498db"))
            patch.set_alpha(0.75)

        title_suffix = f" ({fee_val} FEE)" if fee_val > 0 else ""
        ax.set_title(f"Carter Proximity Cost Distribution on {inst}{title_suffix}")
        ax.set_ylabel("Feasible Proximity Cost f(x)")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()

        p_out = out_dir / f"boxplot_{inst}.png"
        fig.savefig(p_out, bbox_inches="tight")
        plt.close(fig)
        saved.append(p_out)

    # Quality vs Speed trade-off chart
    if not feas_df.empty:
        summary = feas_df.groupby("algorithm", as_index=False).agg(
            mean_f=("best_f", "mean"), avg_time=("elapsed", "mean")
        )
        fig, ax1 = plt.subplots(figsize=(10, 5.2))
        ax2 = ax1.twinx()

        colors = [SOLVER_PALETTE.get(a, "#3498db") for a in summary["algorithm"]]
        ax1.bar(summary["algorithm"], summary["mean_f"], color=colors, alpha=0.75)
        ax1.set_ylabel("Mean Feasible Proximity Cost f(x)", color="#1f618d")
        ax1.tick_params(axis="x", rotation=25)

        ax2.plot(
            summary["algorithm"],
            summary["avg_time"],
            color="#c0392b",
            marker="o",
            linewidth=2.0,
        )
        ax2.set_ylabel("Mean Runtime (seconds)", color="#c0392b")

        ax1.set_title("Solution Quality vs. Runtime Trade-off Across Solvers")
        fig.tight_layout()
        q_path = out_dir / "quality_vs_speed.png"
        fig.savefig(q_path, bbox_inches="tight")
        plt.close(fig)
        saved.append(q_path)

    return saved


if __name__ == "__main__":
    paths = generate_plots()
    print(f"Generated {len(paths)} figures in results/figures/.")
