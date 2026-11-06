"""Consistent matplotlib styling and color palette for project figures (`analysis/style.py`)."""

from __future__ import annotations

import matplotlib.pyplot as plt

SOLVER_PALETTE: dict[str, str] = {
    "random": "#7f8c8d",
    "dsatur": "#95a5a6",
    "sa": "#e67e22",
    "exact": "#2c3e50",
    "ga": "#3498db",
    "ga_memetic": "#1f618d",
    "pso": "#2ecc71",
    "pso_memetic": "#1e8449",
    "aco": "#9b59b6",
    "aco_memetic": "#6c3483",
}


def apply_plot_style() -> None:
    """Apply clean publication-grade matplotlib rcParams."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )
