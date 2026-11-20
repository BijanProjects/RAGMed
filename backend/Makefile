PYTHON ?= python3.12
VENV   := .venv
BIN    := $(VENV)/bin

.DEFAULT_GOAL := help
.PHONY: help venv install lock lint format typecheck test check clean

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

venv: ## Create the virtual environment
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --quiet --upgrade pip

install: venv ## Install locked dev dependencies and git hooks
	$(BIN)/pip install --quiet -r requirements/dev.txt
	cd .. && backend/$(BIN)/pre-commit install

lock: venv ## Re-resolve requirements/dev.txt from requirements/dev.in
	$(BIN)/pip-compile --quiet --strip-extras --no-emit-index-url \
		--output-file requirements/dev.txt requirements/dev.in
	sed -i.bak 's#^-e file:///.*/packages/#-e ./packages/#' requirements/dev.txt && rm requirements/dev.txt.bak

lint: ## Lint and check formatting
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: ## Apply formatting and safe lint fixes
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

typecheck: ## Run mypy in strict mode
	$(BIN)/mypy

test: ## Run tests with coverage
	$(BIN)/pytest --cov

check: lint typecheck test ## Everything CI runs

clean: ## Remove caches and build output
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage coverage.xml htmlcov
	find . -name __pycache__ -type d -prune -not -path './$(VENV)/*' -exec rm -rf {} +
