# MinorSafe Remote Server Runbook (`teen_12_18`)

Step-by-step operator guide for running the MinorSafe benchmark on a high-compute remote Linux server (single GPU or multi-GPU node, or SLURM cluster).

---

## 1. Environment Bootstrap & Pre-Flight Verification

```bash
# 1. Bootstrap Python 3.11 + uv virtual environment
bash scripts/setup_server.sh
source .venv/bin/activate

# 2. Authenticate with HuggingFace for gated models (Llama-3.1/3.2, Gemma-2, ShieldGemma)
export HF_TOKEN="hf_your_token_here"
export HF_HOME="/scratch/$USER/hf_cache"

# 3. Inspect hardware and dry-run model downloads
minorsafe info --profile gpu_80gb
python scripts/download_models.py --profile gpu_80gb --dry-run

# 4. Pre-download and verify all checkpoints before launching GPU jobs
python scripts/download_models.py --profile gpu_80gb --verify

# 5. Run the CPU smoke test and full test suite
make test
make smoke
```

---

## 2. Building the Dataset & Contrastive Corpus

```bash
# Builds data/processed/examples.parquet, data/processed/pairs.parquet, and results/data_card.md
make data PROFILE=gpu_80gb
```

- **Locked Age Group**: `12–18` (`teen_12_18`, target Flesch-Kincaid grade $\le 8.0$).
- **Unicode & Emoji Policy**: `NFC` normalization only; all emojis (`💀`, `😭`, `🔥`, `🖕`), symbols, and special characters are preserved end-to-end.

---

## 3. Running the Core Benchmark Suite

All core experiments exclude multi-turn flow detection by default (`make main`):

```bash
# Run core experiments (main_grid, recomposition, subculture_audit, ramp_up, quant_grid)
make main PROFILE=gpu_80gb

# Or run individual phases inside tmux / screen:
bash scripts/run_phase.sh main_grid gpu_80gb
bash scripts/run_phase.sh recomposition gpu_80gb
bash scripts/run_phase.sh subculture_audit gpu_80gb
bash scripts/run_phase.sh ramp_up gpu_80gb
bash scripts/run_phase.sh quant_grid gpu_80gb
```

### Optional Multi-Turn Extension (Non-Core)
```bash
make multiturn PROFILE=gpu_80gb
```

### SLURM Job Array Submission
```bash
sbatch --export=ALL,PROFILE=gpu_80gb scripts/slurm/run_grid.sbatch
```

---

## 4. Human Annotation Calibration (150 Samples)

```bash
# Export 150 stratified, method-blinded responses to results/human_eval_150.csv
minorsafe judge export-human --n 150 --out results/human_eval_150.csv

# After labeling human_* columns in results/human_eval_150_labeled.csv:
minorsafe judge agreement --csv results/human_eval_150_labeled.csv --out results/human_judge_agreement.json
```

---

## 5. Aggregation, Raw Token/Time Cost Table, Figures & Executive Report

```bash
make analyze
```

Outputs in `results/`:
- `results/cost_table.csv` — Raw token counts (`tok_total_mean`, `setup_tokens`) and wall-clock/GPU execution times (`e2e_ms_p50`, `decode_tok_per_s_mean`, `setup_wall_s`, `setup_gpu_s`) with **no hardcoded dollar prices**.
- `results/figures/fig1_*.pdf` .. `fig8_*.pdf` — All 8 publication figures.
- `results/tables/main_benchmark.{md,tex}` & `results/tables/hypothesis_verdicts.{md,tex}`.
- `results/REPORT.md` — Executive report with hypothesis verdicts `H1`–`H10`.

