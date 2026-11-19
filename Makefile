.PHONY: setup setup-gpu lint format test test-integration smoke data preflight main rampup guards \
	main-grid recomposition subculture quant multiturn analyze remote-check remote-setup \
	remote-sync remote-run remote-status remote-logs remote-pull

UV ?= uv
# Hardware profile for GPU phases: a name in configs/hardware/ or "auto" (detect via nvidia-smi).
PROFILE ?= auto
PHASE ?= smoke

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
	$(UV) run minorsafe data build

preflight:
	$(UV) run minorsafe preflight --hardware $(PROFILE)

# Core single-turn benchmark (multiturn and quant are excluded by design; see plan.md Decision 5).
main: data rampup guards main-grid recomposition subculture analyze

rampup:
	bash scripts/run_phase.sh rampup $(PROFILE)

guards:
	bash scripts/run_phase.sh guards $(PROFILE)

main-grid:
	bash scripts/run_phase.sh main $(PROFILE)

recomposition:
	bash scripts/run_phase.sh recomposition $(PROFILE)

subculture:
	bash scripts/run_phase.sh subculture $(PROFILE)

quant:
	bash scripts/run_phase.sh quant $(PROFILE)

multiturn:
	bash scripts/run_phase.sh multiturn $(PROFILE)

analyze:
	$(UV) run minorsafe analyze

# Automated remote server orchestration over University VPN (reads gitignored .remote.env)
remote-check:
	bash scripts/remote.sh check

remote-sync:
	bash scripts/remote.sh sync

remote-setup:
	bash scripts/remote.sh setup

remote-run:
	bash scripts/remote.sh run $(PHASE) $(PROFILE)

remote-status:
	bash scripts/remote.sh status

remote-logs:
	bash scripts/remote.sh logs

remote-pull:
	bash scripts/remote.sh pull
