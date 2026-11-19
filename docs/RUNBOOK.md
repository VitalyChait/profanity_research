# MinorSafe Remote Server Runbook (`teen_12_18`)

Step-by-step operator guide for running the MinorSafe benchmark on a remote Linux GPU server behind a University VPN (both **automated** from your local machine and **manual** directly on the server). Every CLI command below is verified by `tests/unit/test_cli_entrypoints.py`.

> [!WARNING]
> **Placeholder components fail closed.** GPU profiles set `allow_placeholders: false`, so any phase that still depends on a CPU placeholder raises `PlaceholderBackendError` instead of producing numbers. Currently gated: M1 guards (`m1_encoder_guard`, `m1_decoder_guard`), M2 LoRA (`m2_lora`), the `mlm_substitute` action, `capability_eval`, the LLM judge (`judge_backend: vllm|hf`), and reference-LM perplexity (`ppl_model`). M0a/M0b and M3 steering (SwFC/StTP/StMP/Ortho, and the probe guard with a non-MLM action) are real. See `src/minorsafe/placeholders.py`.

---

## 0. Automated Remote Execution Over University VPN (Local -> Server)

Connect to your University VPN first, then configure local gitignored credentials (`.remote.env` is ignored by `.gitignore` and protected by `scripts/install_git_hooks.sh`):

```bash
# 1. Create gitignored credentials file (never committed to this public repo)
cp .remote.env.example .remote.env
chmod 600 .remote.env
# Edit .remote.env: set REMOTE_HOST, REMOTE_USER="root", REMOTE_DIR, HF_TOKEN, etc.

# 2. Install local pre-commit hook that blocks accidental credential commits
bash scripts/install_git_hooks.sh

# 3. Verify VPN reachability, GPU idle status, and that no other user is logged in (`who`)
make remote-check

# 4. Install SSH key (so root password/sshpass isn't needed afterwards), sync repo, and bootstrap GPU env
bash scripts/remote.sh setup-ssh-key
make remote-setup

# 5. Run a phase remotely (or --detach inside tmux on the server) and monitor smart logs
bash scripts/remote.sh run data auto
bash scripts/remote.sh run main gpu_80gb --detach
make remote-status
make remote-logs
make remote-pull
```

`scripts/remote.sh` enforces `.remote.env` gitignore/untracked status and `chmod 600`, checks TCP reachability within `VPN_CHECK_TIMEOUT`, validates server exclusivity (`REQUIRE_EXCLUSIVE=1` aborts if other users or GPU compute apps are active), excludes `.remote.env*`/`.secrets/`/`*.pem`/`*.key` from `rsync`, and writes smart JSONL + ETA logs remotely under `results/logs/`.

---

## 1. Manual Activation & Pre-Flight Verification (Directly on Server or Local)

If you SSH onto the server manually (or work locally), use `scripts/activate.sh` to load `.remote.env`, activate `.venv`, install git hooks, and run all initial hardware/environment/exclusivity/secret checks:

```bash
# Manual activation + preflight check
source scripts/activate.sh auto
minorsafe preflight --hardware auto

# First-time manual bootstrap (when not using `make remote-setup`)
export HF_TOKEN="hf_your_token_here"          # gated models: Llama-3.x, Gemma
export HF_HOME="/scratch/$USER/hf_cache"      # fast, persistent storage for checkpoints
bash scripts/setup_server.sh
source scripts/activate.sh auto

# Inspect hardware and pre-download checkpoints
minorsafe info --hardware auto
python scripts/download_models.py --profile auto --dry-run
python scripts/download_models.py --profile auto --verify

# CPU test suite + CPU smoke pipeline (dummy models; validates plumbing only)
make test
make smoke
```

Available profiles in `configs/hardware/`: `cpu_debug`, `gpu_24gb`, `gpu_48gb`, `gpu_80gb`, `multi_gpu_80gb`. Every command that takes a profile accepts `--hardware <name|auto>` (alias `--profile`).

---

## 2. Building the Dataset & Contrastive Corpus

```bash
make data                  # == minorsafe data build  (add --offline to use only the seed corpus)
minorsafe data report      # prints results/data_card.md
```

- Writes `data/processed/examples.parquet`, `data/processed/pairs.parquet`, and `results/data_card.md`.
- **Locked Age Group**: `12–18` (`teen_12_18`, target Flesch-Kincaid grade ≤ 8.0).
- **Unicode & Emoji Policy**: `NFC` normalization only; all emojis, symbols, and special characters are preserved end to end.

---

## 3. Running the Benchmark & Smart Logging

Each phase is run by `scripts/run_phase.sh <phase> [profile]`, which:
1. Runs `minorsafe preflight --hardware <profile>` before GPU phases (verifying CUDA/VRAM, server exclusivity, disk space, and credential safety; pass `MINORSAFE_ALLOW_BUSY=1` or `--allow-busy` to override exclusivity).
2. Streams output to `results/logs/<phase>_<timestamp>.log` and symlinks `results/logs/latest.log`.
3. Samples GPU utilization/VRAM/temperature every 15s into `results/logs/<phase>_<timestamp>.gpu.csv`.
4. Emits batch-level ETA lines and updates `results/logs/live_status.json` + `results/logs/events.jsonl`.
5. Adds `--final` automatically only for experiments whose config targets the `test` split. Use `DRY_RUN=1` to print the command without running it.

| Phase | Experiment config | Split | In `make main` |
| :--- | :--- | :--- | :--- |
| `smoke` | CPU smoke pipeline (dummy models) | – | – |
| `data` | `configs/data.yaml` | – | ✅ |
| `rampup` | `ramp_up.yaml` | dev | ✅ |
| `guards` | `guards.yaml` | test | ✅ |
| `main` | `main_grid.yaml` | test | ✅ |
| `recomposition` | `recomposition.yaml` | test | ✅ |
| `subculture` | `subculture_audit.yaml` | test | ✅ |
| `quant` | `quant_grid.yaml` | dev | – (secondary) |
| `multiturn` | `multiturn.yaml` | dev | – (optional, Decision 5) |
| `analyze` | aggregates `results/runs/` | – | ✅ |

```bash
# Full core pipeline (data -> rampup -> guards -> main -> recomposition -> subculture -> analyze)
make main PROFILE=gpu_80gb

# Or single phases:
make guards PROFILE=gpu_80gb                  # == bash scripts/run_phase.sh guards gpu_80gb
bash scripts/run_phase.sh main gpu_80gb
DRY_RUN=1 bash scripts/run_phase.sh main gpu_80gb   # show the command only

# Secondary / optional studies
make quant PROFILE=gpu_80gb
make multiturn PROFILE=gpu_80gb

# Live progress + completed shard counts
minorsafe status
```

### SLURM Job Array

```bash
mkdir -p results/logs   # SLURM does not create the --output directory
sbatch --export=ALL,PROFILE=gpu_80gb scripts/slurm/run_grid.sbatch
```

Array tasks `0-5` map to phases `main guards recomposition subculture rampup quant`. Run `make data` before submitting, and `make analyze` after all tasks finish.

---

## 4. Human Annotation Calibration (150 Samples)

```bash
# Export 150 stratified, method-blinded responses (companion key saved separately as *_key.csv)
minorsafe judge export-human --n 150 --out results/human_eval_150.csv

# After filling the human_* columns and saving as results/human_eval_150_labeled.csv:
minorsafe judge agreement --human-csv results/human_eval_150_labeled.csv --out results/human_judge_agreement.json
```

---

## 5. Aggregation, Raw Token/Time Cost Table, Figures & Report

```bash
make analyze      # == minorsafe analyze [--runs results/runs] [--out results]
```

Outputs in `results/`:
- `cost_table.csv`: raw token counts (`tok_total_mean`, `setup_tokens`) and wall-clock/GPU times (`e2e_ms_p50`, `decode_tok_per_s_mean`, `setup_wall_s`, `setup_gpu_s`), with **no hardcoded dollar prices**. `energy_j_mean` and `peak_vram_mb` are populated on CUDA via `GPUSampler` (`NaN` on CPU).
- `figures/fig1_*.pdf` .. `fig8_*.pdf`
- `tables/main_benchmark.{md,tex}` and `tables/hypothesis_verdicts.{md,tex}`: H1–H10 verdicts computed by explicit decision rules (`src/minorsafe/analysis/hypotheses.py`); `NOT_EVALUATED` entries state the missing data.
- `REPORT.md`

---

## 6. Troubleshooting

| Symptom | Cause / Fix |
| :--- | :--- |
| `Cannot reach <host>:<port>` | Ensure University VPN is connected and verify `REMOTE_HOST` / `REMOTE_PORT` in `.remote.env`. |
| `Server/GPU is not exclusive` / `minorsafe preflight` fails | Another user is logged in (`who`), a foreign GPU process is running, or `.minorsafe.lock` is active. Run `make remote-check` to inspect; override only intentionally with `--allow-busy` or `MINORSAFE_ALLOW_BUSY=1`. |
| `PlaceholderBackendError: ... is a CPU placeholder` | Expected on GPU profiles until the component is implemented. Remove that method/flag from the experiment, or implement the real backend. Never set `allow_placeholders: true` on a profile used for reported results. |
| `PermissionError: ... targets the 'test' split` | Run via `run_phase.sh` (adds `--final` for test-split configs) or pass `--final` explicitly. |
| `ValueError: No calibration pairs for steering concept ...` | The data build produced no pairs for a configured concept; check `results/data_card.md` and the method's `concepts`. |
| Gated repo / 401 errors | `export HF_TOKEN=...` (or set in `.remote.env`) and accept the model licence on huggingface.co. |
| CUDA OOM | Use a smaller profile (`gpu_24gb` uses QLoRA for L models) or lower `batch_size` via `-o batch_size=1`. |
| Pull results locally | `make remote-pull` (or `bash scripts/remote.sh pull`). |
