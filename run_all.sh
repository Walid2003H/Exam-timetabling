#!/usr/bin/env bash
set -e

if [[ "${1:-}" == "--quick" ]]; then
    echo "Running contract tests..."
    if python3 -c "import pytest" >/dev/null 2>&1; then
        python3 -m pytest tests/
    else
        PYTHONPATH=. python3 tests/test_contract.py
    fi
    echo "Quick contract check completed."
else
    echo "Running current contract checks (the experiment pipeline is added in later phases)..."
    if python3 -c "import pytest" >/dev/null 2>&1; then
        python3 -m pytest tests/
    else
        PYTHONPATH=. python3 tests/test_contract.py
    fi
fi
