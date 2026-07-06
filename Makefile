.PHONY: setup lint typecheck test check backtest repro clean

setup:            ## Install pinned deps into .venv
	uv sync --frozen

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:              ## Auto-fix lint + formatting
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck test   ## Everything CI runs

backtest:         ## Walk-forward backtest -> reports/ (Phase 5)
	uv run engine backtest

repro: backtest   ## Regenerate every number cited in the README from scratch

clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache .coverage
