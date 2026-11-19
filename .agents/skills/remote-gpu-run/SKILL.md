---
name: remote-gpu-run
description: >-
  Run, verify, and troubleshoot the MinorSafe benchmark on a remote Linux server
  with CUDA GPUs or SLURM. Use when setting up a GPU server, pre-downloading
  HuggingFace models, running experiment phases, or debugging GPU profile failures.
---

# Running MinorSafe on a Remote Linux GPU Server

Use these verified commands instead of stale targets in `Makefile` or `docs/RUNBOOK.md`.

## 1. Environment Bootstrap (CUDA 12.x + uv)

`scripts/setup_server.sh` creates the Python 3.11 `.venv` and directories, but only installs `.[dev]` (CPU/base). Follow it with `make setup-gpu` (or `uv sync`) to install CUDA `torch`, `bitsandbytes`, and `vllm`:

```bash
bash scripts/setup_server.sh
uv sync --extra gpu --extra vllm --extra dev
source .venv/bin/activate
```

Set HuggingFace credentials and cache path (use fast scratch/persistent storage for checkpoints):

```bash
export HF_TOKEN="hf_your_token_here"
export HF_HOME="/scratch/$USER/hf_cache"
export TOKENIZERS_PARALLELISM="false"
```

## 2. Pre-Flight Verification & Model Download

Available GPU profiles in `configs/hardware/`: `gpu_24gb`, `gpu_48gb`, `gpu_80gb`, `multi_gpu_80gb`.

```bash
minorsafe info --profile gpu_80gb
python scripts/download_models.py --profile gpu_80gb --dry-run
python scripts/download_models.py --profile gpu_80gb --verify
make test
minorsafe smoke
```

## 3. Dataset Build

```bash
bash scripts/run_phase.sh data
# Equivalent: minorsafe data
```

## 4. Running Experiments

### Single Node (`tmux` / `screen`)

Run phases via `scripts/run_phase.sh <phase> <profile>` (`main_grid`, `recomposition`, `subculture_audit`, `ramp_up`, `quant_grid`, `multiturn`, `analyze`):

```bash
bash scripts/run_phase.sh main_grid gpu_80gb
bash scripts/run_phase.sh recomposition gpu_80gb
bash scripts/run_phase.sh subculture_audit gpu_80gb
bash scripts/run_phase.sh ramp_up gpu_80gb
bash scripts/run_phase.sh quant_grid gpu_80gb
```

### SLURM Job Array

`scripts/slurm/run_grid.sbatch` runs array tasks `0-4` (`main_grid`, `recomposition`, `subculture_audit`, `ramp_up`, `quant_grid`):

```bash
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

- **Broken `Makefile` targets (do not use)**:
  - `make smoke` passes `--hardware cpu_debug` (invalid flag for `minorsafe smoke`). Use `minorsafe smoke`.
  - `make data` runs `minorsafe data build --hardware auto` (invalid subcommand/flag). Use `minorsafe data` or `bash scripts/run_phase.sh data`.
  - `make main` invokes `rampup`, `guards`, `main`, `subculture`, which do not exist in `scripts/run_phase.sh`.
- **Fail-closed placeholder guard (`src/minorsafe/placeholders.py`)**: GPU profiles set `allow_placeholders: false` (default). Running `capability_eval` (`main_grid.yaml`) or placeholder-backed methods (`m1_encoder_guard`, `m2_lora`, `mlm_substitute` action) raises `PlaceholderBackendError` until real GPU implementations replace those placeholders.

