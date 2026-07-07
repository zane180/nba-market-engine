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

backtest:         ## Walk-forward backtest -> reports/
	uv run engine backtest

repro:            ## Regenerate every number cited in the README from scratch
	uv run engine report pregame
	uv run engine report live
	uv run engine backtest

PLIST = $(HOME)/Library/LaunchAgents/com.nba-market-engine.paper.plist

install-paper-daemon:   ## launchd agent: run the paper loop continuously
	mkdir -p data/logs $(HOME)/Library/LaunchAgents
	sed -e "s|__REPO__|$(CURDIR)|g" -e "s|__UV__|$$(command -v uv)|g" \
		deploy/com.nba-market-engine.paper.plist.template > $(PLIST)
	launchctl unload $(PLIST) 2>/dev/null || true
	launchctl load $(PLIST)
	@echo "installed; logs: data/logs/paper.log"

uninstall-paper-daemon:
	launchctl unload $(PLIST) 2>/dev/null || true
	rm -f $(PLIST)
	@echo "removed"

clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache .coverage
