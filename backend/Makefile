PYTHON ?= python3.12
VENV   := .venv
BIN    := $(VENV)/bin

.DEFAULT_GOAL := help
.PHONY: help venv install install-bench lock lint format typecheck test check clean search report significance export-onnx benchmark-all mcp-ingest mcp-serve

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

venv: ## Create the virtual environment
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --quiet --upgrade pip

install: venv ## Install locked dev dependencies and git hooks
	$(BIN)/pip install --quiet -r requirements/dev.txt
	cd .. && backend/$(BIN)/pre-commit install

install-bench: install ## Add the embedding benchmark dependencies (PyTorch, ~1 GB)
	$(BIN)/pip install --quiet -r requirements/bench.txt

lock: venv ## Re-pin requirements/*.txt from requirements/*.in
	@# prod.txt reuses dev.txt's versions; pip refuses editable lines in a constraints file.
	$(BIN)/pip-compile --quiet --strip-extras --no-emit-index-url \
		--output-file requirements/dev.txt requirements/dev.in
	grep -v '^-e ' requirements/dev.txt > requirements/constraints.txt
	for name in dev bench prod; do \
		$(BIN)/pip-compile --quiet --strip-extras --no-emit-index-url \
			--output-file requirements/$$name.txt requirements/$$name.in && \
		sed -i.bak -E 's#^-e file:///.*/backend/(packages|services)/#-e ./\1/#' requirements/$$name.txt && \
		rm requirements/$$name.txt.bak; \
	done
	@# Lock files must work on any machine: fail if a local absolute path slipped through.
	@! grep -n "file://" requirements/*.txt || (echo "absolute paths in lock files" && false)

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

Q ?= What are the symptoms of asthma?
search: ## Ask the knowledge base a question on CPU with BGE-M3:  make search Q="..."
	$(BIN)/python -m ragmed_retrieval search "$(Q)"

export-onnx: ## Export BGE-M3 to int8 ONNX for fast CPU search (needs install-bench)
	$(BIN)/python -m ragmed_retrieval export-onnx

benchmark-all: ## Re-run every experiment (~1 h on a GPU); progress: tail -f data/artifacts/benchmark.log
	evals/run_all.sh

VECTORS ?= $(firstword $(wildcard data/artifacts/embeddings/BAAI__bge-m3-*.npy))
mcp-ingest: ## Load the knowledge base into the local vector store (needs make search or benchmark first)
	$(BIN)/python -m mcp_medical ingest $(if $(VECTORS),--vectors $(VECTORS))

mcp-serve: ## Run the Medical Knowledge MCP server on http://127.0.0.1:9001/mcp
	$(BIN)/python -m mcp_medical serve

report: ## Print the embedding benchmark results table
	$(BIN)/python -m ragmed_retrieval report

significance: ## Recompute bootstrap confidence intervals for the benchmark
	$(BIN)/python evals/analysis.py

clean: ## Remove caches and build output
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage coverage.xml htmlcov
	find . -name __pycache__ -type d -prune -not -path './$(VENV)/*' -exec rm -rf {} +
