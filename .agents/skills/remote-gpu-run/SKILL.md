---
name: remote-gpu-run
description: >-
  Run, verify, and troubleshoot the MinorSafe benchmark on a remote Linux server
  with CUDA GPUs or SLURM. Use when setting up a GPU server, pre-downloading
  HuggingFace models, running experiment phases, or debugging GPU profile failures.
---

# Running MinorSafe on a Remote Linux GPU Server

Supports both **automated execution over University VPN** (from your local workstation via gitignored `.remote.env`) and **manual activation** directly on the server. See `docs/RUNBOOK.md` for full details.

## 0. Automated Remote Execution Over University VPN

Connect to the University VPN, then create your gitignored `.remote.env` (never committed to this public repo) and install the pre-commit secret guard:

```bash
cp .remote.env.example .remote.env
chmod 600 .remote.env
# Edit .remote.env, connect to University VPN, then run the repo entry point:
./setup.sh                                    # (or `make bootstrap`) hooks + local .venv + SSH key + check + remote setup

make remote-check                             # VPN TCP check + exclusivity (`who` + GPU idle)
bash scripts/remote.sh setup-ssh-key          # installs SSH key (uses OpenSSH askpass if REMOTE_PASSWORD set; no sshpass needed)
make remote-setup                             # rsync + setup_server.sh + remote preflight
bash scripts/remote.sh run main gpu_80gb --detach
make remote-status                            # live ETA status, tmux, GPUs, latest log tail
make remote-logs                              # follow results/logs/latest.log
make remote-pull                              # rsync results/ back (excludes *.safetensors, *.pt)
```

## 1. Manual Activation & Bootstrap (On Server)

```bash
# Quick manual activation + preflight checks (sources .remote.env if present, activates .venv)
source scripts/activate.sh auto

# First-time bootstrap (installs uv if missing, runs `uv sync --extra gpu --extra vllm --extra dev`, runs info)
export HF_TOKEN="hf_your_token_here"
export HF_HOME="/scratch/$USER/hf_cache"
bash scripts/setup_server.sh
source scripts/activate.sh auto
```

## 2. Pre-Flight Verification & Model Download

Available GPU profiles in `configs/hardware/`: `gpu_24gb`, `gpu_48gb`, `gpu_80gb`, `multi_gpu_80gb`.

```bash
minorsafe preflight --hardware gpu_80gb
minorsafe info --profile gpu_80gb
python scripts/download_models.py --profile gpu_80gb --dry-run
python scripts/download_models.py --profile gpu_80gb --verify
make test
minorsafe smoke
```

## 3. Dataset Build

```bash
bash scripts/run_phase.sh data
# Equivalent: minorsafe data build (or make data)
```

## 4. Running Experiments (with Preflight & Smart Logging)

`scripts/run_phase.sh <phase> <profile>` runs `minorsafe preflight`, logs batch ETAs + `live_status.json` + `events.jsonl`, and samples GPU telemetry to `results/logs/*.gpu.csv`.

```bash
bash scripts/run_phase.sh main gpu_80gb
bash scripts/run_phase.sh guards gpu_80gb
bash scripts/run_phase.sh recomposition gpu_80gb
bash scripts/run_phase.sh subculture gpu_80gb
bash scripts/run_phase.sh rampup gpu_80gb
bash scripts/run_phase.sh tomography gpu_80gb
bash scripts/run_phase.sh quant gpu_80gb
```

### SLURM Job Array

`scripts/slurm/run_grid.sbatch` runs array tasks `0-5` (`main guards recomposition subculture rampup quant`):

```bash
mkdir -p results/logs
sbatch --export=ALL,PROFILE=gpu_80gb scripts/slurm/run_grid.sbatch
```

## 5. Human Calibration & Analysis

```bash
minorsafe judge export-human --n 150 --out results/human_eval_150.csv
# After labeling:
minorsafe judge agreement --csv results/human_eval_150_labeled.csv --out results/human_judge_agreement.json
bash scripts/run_phase.sh analyze
```

## Known Repo Gotchas

- **Aligned `Makefile` / CLI / `run_phase.sh` targets**: `make smoke`, `make data`, `make preflight`, `make main`, `make remote-*`, and `scripts/run_phase.sh` phase aliases (`rampup`/`ramp_up`, `guards`, `main`/`main_grid`, `recomposition`, `subculture`/`subculture_audit`, `quant`/`quant_grid`, `tomography`, `multiturn`, `analyze`) are aligned and verified by `tests/unit/test_cli_entrypoints.py`.
- **Exclusivity guard**: CUDA runs (`run_experiment` and `run_phase.sh`) abort if another user is logged in (`who`), a foreign GPU compute process is active, or `.minorsafe.lock` is held. Pass `--allow-busy` to `preflight` or set `MINORSAFE_ALLOW_BUSY=1` (`REQUIRE_EXCLUSIVE=0` in `.remote.env`) only when intentionally sharing.
- **GPU backends & placeholder guard (`src/minorsafe/placeholders.py`)**: All benchmark methods (`m1_encoder_guard`, `m1_decoder_guard`, `MLMRecomposer`, `m2_lora`, `m3_steer_*`, `tomography`), evaluators (`LLMJudge`, `BERTScorer`, reference-LM PPL, `tinyMMLU`/`tinyGSM8K` capability retention), and teacher synthesis use real GPU implementations on `gpu_*` profiles. Placeholder fallbacks (`allow_placeholders: true`) are strictly restricted to `cpu_debug` tests and raise `PlaceholderBackendError` if called on GPU profiles.
