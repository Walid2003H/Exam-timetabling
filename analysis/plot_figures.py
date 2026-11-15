"""Backward-compatible wrapper forwarding to `analysis.make_figures`."""

from __future__ import annotations

from analysis.make_figures import generate_plots

if __name__ == "__main__":
    paths = generate_plots()
    print(f"Generated {len(paths)} figures in results/figures/.")
