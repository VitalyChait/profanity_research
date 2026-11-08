.PHONY: setup setup-gpu lint format test test-integration smoke data main analyze

UV ?= uv

setup:
	$(UV) sync --extra cpu --extra dev

setup-gpu:
	$(UV) sync --extra gpu --extra vllm --extra dev

lint:
	$(UV) run ruff check src tests scripts
	$(UV) run ruff format --check src tests scripts

format:
	$(UV) run ruff check --fix src tests scripts
	$(UV) run ruff format src tests scripts

test:
	$(UV) run pytest -m "unit"

test-integration:
	$(UV) run pytest -m "integration"

smoke:
	$(UV) run minorsafe smoke --hardware cpu_debug

data:
	$(UV) run minorsafe data build --hardware auto

main:
	bash scripts/run_phase.sh data
	bash scripts/run_phase.sh rampup
	bash scripts/run_phase.sh guards
	bash scripts/run_phase.sh main
	bash scripts/run_phase.sh recomposition
	bash scripts/run_phase.sh subculture
	bash scripts/run_phase.sh analyze

analyze:
	$(UV) run minorsafe analyze
