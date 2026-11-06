#!/usr/bin/env bash
set -e

if [[ "${1:-}" == "--quick" ]]; then
    echo "[1/4] Running unit test suites..."
    if python3 -c "import pytest" >/dev/null 2>&1; then
        python3 -m pytest tests/ -q
    else
        PYTHONPATH=. python3 tests/test_contract.py
    fi
    echo "[2/4] Running quick hyperparameter tuning check..."
    PYTHONPATH=. python3 -m experiments.tune --quick
    echo "[3/4] Running quick end-to-end benchmark & ablation suite..."
    PYTHONPATH=. python3 -m experiments.run --quick
    echo "[4/4] Running statistical analysis & figure generation..."
    PYTHONPATH=. python3 -m analysis.stats
    PYTHONPATH=. python3 -m analysis.make_figures
    echo "Quick end-to-end pipeline completed successfully."
else
    echo "[1/4] Running unit test suites..."
    if python3 -c "import pytest" >/dev/null 2>&1; then
        python3 -m pytest tests/ -q
    else
        PYTHONPATH=. python3 tests/test_contract.py
    fi
    echo "[2/4] Running hyperparameter tuning on tune-* instances..."
    PYTHONPATH=. python3 -m experiments.tune
    echo "[3/4] Running benchmark & ablation suite..."
    PYTHONPATH=. python3 -m experiments.run --mode pilot
    echo "[4/4] Running statistical analysis & figure generation..."
    PYTHONPATH=. python3 -m analysis.stats
    PYTHONPATH=. python3 -m analysis.make_figures
    echo "Full experiment pipeline completed successfully."
fi
