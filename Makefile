# Shortcuts. Every target is a plain mlt / uv command, so nothing here is required.
# Usage: make run CONFIG=configs/examples/adult.yaml
CONFIG ?= configs/examples/breast_cancer.yaml
EXTRAS ?= all,dev

.PHONY: install install-cpu folds run check tune test lint format freeze demo clean

install:            ## create .venv and install mlt with extras (EXTRAS=all,dev)
	uv venv
	uv pip install -e ".[$(EXTRAS)]"

install-cpu:        ## same, with CPU-only PyTorch wheels (smaller, no CUDA)
	uv venv
	uv pip install --torch-backend cpu -e ".[$(EXTRAS)]"

folds:
	uv run mlt folds $(CONFIG)

run:
	uv run mlt run $(CONFIG)

check:
	uv run mlt check $(CONFIG)

tune:
	uv run mlt tune $(CONFIG)

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

freeze:             ## pin the exact environment of this competition
	uv pip freeze > requirements.lock.txt

demo:               ## offline end-to-end check: tabular binary + regression
	uv run mlt example breast_cancer
	uv run mlt run configs/examples/breast_cancer.yaml
	uv run mlt example diabetes
	uv run mlt run configs/examples/diabetes.yaml

clean:
	rm -rf outputs .pytest_cache .ruff_cache
