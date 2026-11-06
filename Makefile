# Common tasks. `make check` is what CI runs (minus the Docker and interop jobs).
PY ?= python
.PHONY: help install test lint check golden golden-check results-index interop evals sandbox clean

help:
	@echo "make install       editable install with dev + serve extras"
	@echo "make test          pytest (no model calls; Docker tests skip without the sandbox image)"
	@echo "make lint          ruff"
	@echo "make check         lint + tests + schema check of the golden trajectories"
	@echo "make golden        regenerate tests/golden (only for a deliberate behaviour change; review the diff)"
	@echo "make results-index regenerate results/README.md"
	@echo "make interop       MCP interop against the official SDK (needs .venv-sdk, see scripts/mcp_interop.py)"
	@echo "make evals         baseline + feature arms on flash (needs DEEPSEEK_API_KEY, about \$$1-2)"
	@echo "make sandbox       build the Docker sandbox image"

install:
	$(PY) -m pip install -e ".[dev,serve]"

test:
	$(PY) -m pytest -q

lint:
	ruff check .

golden-check:
	miniagent-eval check-trajectories tests/golden

check: lint test golden-check

golden:
	$(PY) scripts/make_golden.py
	miniagent-eval replay tests/golden

results-index:
	$(PY) scripts/index_results.py

interop:
	.venv-sdk/bin/python scripts/mcp_interop.py

evals:
	scripts/run_feature_evals.sh

sandbox:
	miniagent-eval sandbox-build

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
