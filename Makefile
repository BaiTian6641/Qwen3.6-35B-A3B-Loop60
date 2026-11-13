PY ?= .venv/bin/python
RUFF ?= .venv/bin/ruff

.PHONY: venv lint ast test check fmt

venv:
	python3 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]" ruff

lint:
	$(RUFF) check src tests scripts

fmt:
	$(RUFF) check --fix src tests scripts

ast:
	$(PY) scripts/ast_check.py --target 3.9

test:
	$(PY) -m pytest

# what CI / pre-commit should run
check: ast lint test
