.PHONY: install format lint test quick clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements.txt

format:
	$(PYTHON) -m black src tests experiments analysis

lint:
	$(PYTHON) -m ruff check src tests experiments analysis
	$(PYTHON) -m black --check src tests experiments analysis

test:
	$(PYTHON) -m pytest tests/

quick:
	./run_all.sh --quick

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
