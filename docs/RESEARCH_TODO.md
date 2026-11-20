# Research Completion TODO (`MinorSafe`, `teen_12_18`)

Checklist to complete the MinorSafe benchmark, analysis, and final research deliverables from the current project state (as of `2026-11-20`).

---

## 1. Completed Milestones (Baseline State)
- [x] **Core Benchmark & Path A Real GPU Backends Implemented:** All placeholders replaced for `gpu_*` profiles (zero-shot & fine-tuned `EncoderGuard`, logprob `DecoderGuard`, `MLMRecomposer`, PEFT `LoRA`/`QLoRA`, `LLMJudge` via vLLM/HF, `gpt2-large` reference PPL, `BERTScorer`, `tinyMMLU`/`tinyGSM8K` capability eval, teacher synthesis, steering dev sweep, LAT tomography, `H1`–`H10` evaluation rules, and `SmartRunLogger`).
- [x] **Software Verification Passing:**
  - Local `ruff check` + `ruff format --check` pass cleanly (`85 files`).
  - Local offline pytest suite (`105 passed, 2 skipped` GPU tests) and standalone `minorsafe smoke --hardware cpu_debug` pass.
  - Remote preflight (`gpu_80gb` on `spark-52a6`, NVIDIA GB10 121.69 GiB) and `bash scripts/remote.sh gpu-test` (`2 passed` on `qwen25_05b` + `toxic_bert`) pass.
- [x] **Phase 1 — `data` Built on Server (`status=0`, `6,090s`):**
  - `casperhansen/llama-3.3-70b-instruct-awq` generated `6,250` raw pairs (`2,011,049` tokens); strict Detoxify/FK/length/MinHash filtering produced **3,558 examples** (`1,759` calib, `697` dev, `1,102` test) and **3,558 contrastive pairs** (`0` cross-split leakage) in `results/data_card.md`.
- [x] **Phase 2 — `tomography` Completed Across All 13 Generators (`status=0`, `1,386s`):**
  - Meta Llama & Gemma gated access confirmed working; 13 tomography JSON reports + `summary.json` generated on `spark-52a6`.
- [▶] **Phase 3 — `guards` Currently Running on `spark-52a6`:**
  - Tmux session `minorsafe_guards` started at `12:00:00` (`28` runs across `M`-tier models, ETA ~7–8h).

---

## 2. Immediate Operational Hygiene (Local)
- [ ] **Commit New Documentation Files:** Review `git status` and commit [`EXPERIMENT_RATIONALE_AND_METRICS.md`](./EXPERIMENT_RATIONALE_AND_METRICS.md) and [`RESEARCH_TODO.md`](./RESEARCH_TODO.md) cleanly (`.remote.env` remains untracked/gitignored and protected by `.git/hooks/pre-commit`).
- [ ] **Sync Latest `results/tomography/` Locally (after `guards` check):** Run `bash scripts/remote.sh pull` once `guards` finishes so the 13th tomography report (`llama31_8b.json`) and `results/tomography/summary.json` are present locally too.

---

## 3. Remote GPU Experiment Execution (`spark-52a6`, One Phase at a Time)

> Always check `bash scripts/remote.sh status` (`no tmux sessions` and GPU idle) before starting the next phase so the exclusivity lock passes. Every run checkpoints per batch via `RunStore` and resumes safely if interrupted.

- [ ] **Verify `guards` Completion & Canary Metrics (`28` runs, in progress):**
  - Monitor with `bash scripts/remote.sh status` / `bash scripts/remote.sh logs`.
  - Confirm `status=0`, verify `casperhansen/mistral-small-24b-instruct-2501-awq` (`vllm` judge), `roberta-large` BERTScore, and `gpt2-large` PPL finished without `dtype` or OOM issues, and note exact GB10 prompts/sec from `results/logs/guards_*.log`.
- [ ] **Run `subculture` (`104` runs, `test` split, est. ~6–12h):**
  - Command: `bash scripts/remote.sh run subculture gpu_80gb --detach`
  - Unlocks `H5` (`StTP-Ortho` vs. `StTP`/`M0b` teen-slang over-refusal) and obfuscation robustness checks.
- [ ] **Run `recomposition` (`104` runs, `test` split, est. ~6–12h):**
  - Command: `bash scripts/remote.sh run recomposition gpu_80gb --detach`
  - Unlocks `H4` (`StTP` and `mlm_substitute` vs. `llm_rewrite`) and Fig. 6 (`BERTScore`, `SARI`, `FK`).
- [ ] **Run `rampup` (`220` runs, `dev` split, est. ~18–32h):**
  - Command: `bash scripts/remote.sh run rampup gpu_80gb --detach`
  - Produces $N_{\text{calib}} \in \{25, 50, 100, 500, 2000\}$ ramp-up curves (Fig. 4) and `setup_wall_s`/`setup_tokens` inputs for Fig. 3.
- [ ] **Run `main` (`182` runs, `test` split, 3 seeds + capability eval, est. ~45–90h):**
  - Command: `bash scripts/remote.sh run main gpu_80gb --detach`
  - Produces the headline 4-family × 3-scale benchmark (unlocks `H1`, `H2`, `H7`, `H8`, `H9`, Figs. 1–3).
  - *(Note: inspect timing after `guards` + `subculture`; if wall time on GB10 exceeds deadline constraints, consider whether to preserve full `seeds=[0,1,2]` or run per-tier/prioritized subsets via overrides — default full spec has 182 runs × 1,102 × 3 seeds.)*
- [ ] **Run `quant` (`143` runs, `dev` split, secondary study, est. ~15–28h):**
  - Command: `bash scripts/remote.sh run quant gpu_80gb --detach`
  - Unlocks `H10` (INT8 vs. NF4 transferred/native SRI) and Fig. 8.
- [ ] *(Optional — excluded from core `make main` per Decision 5)* **Run `multiturn` (`44` runs, `dev` split):**
  - Command: `bash scripts/remote.sh run multiturn gpu_80gb --detach`

---

## 4. Human Judge Calibration (`Decision 6` — 150 Blinded Samples)
- [ ] **Export 150 Stratified Responses (after `guards` + `main` / core runs are populated):**
  ```bash
  # On server (or locally after pulling results/runs):
  .venv/bin/minorsafe judge export-human --n 150 --out results/human_eval_150.csv
  ```
  - Note: Keep `results/human_eval_150_key.csv` unopened while labeling to preserve blinding.
- [ ] **Annotate All 150 Rows:** Fill `human_age_appropriate`, `human_meaning_preserved`, `human_helpful`, and `human_fluent` (integers `1`–`5` per `configs/prompts.yaml` rubric).
- [ ] **Compute Inter-Annotator Agreement:**
  ```bash
  .venv/bin/minorsafe judge agreement --csv results/human_eval_150_labeled.csv --out results/human_judge_agreement.json
  ```
  - Record quadratic-weighted Cohen's $\kappa$, Spearman's $\rho$, and binary agreement rate for the paper validation section.

---

## 5. Final Aggregation, Sanity Audits & Paper Outputs
- [ ] **Run `analyze` & Pull Everything Locally:**
  ```bash
  bash scripts/remote.sh run analyze gpu_80gb
  bash scripts/remote.sh pull
  ```
- [ ] **Audit Generated Artifacts:**
  - Verify all 8 PDF/PNG figures in `results/figures/` (`fig1`–`fig8`) render legibly without overlapping annotations or empty subplots.
  - Inspect `results/tables/hypothesis_verdicts.md` (and `.tex`) — verify **none of H1–H10 remain `NOT_EVALUATED`**, and inspect cell-level evidence for every `SUPPORTED` / `MIXED` / `NOT_SUPPORTED` verdict.
  - Inspect `results/cost_table.csv` & `results/logs/*.log` (`grep "DONE phase=" results/logs/*.log`) to verify ≥98% non-null eval columns, `setup_tokens`, raw query tokens, wall/GPU seconds, and `DeepSeek-R1` `<think>` accounting.
  - Sanity-check `data_card.md` notes on small subcategories (`slang: 36`, `mature_theme: 44`, `violence: 12`, `drugs: 12`, `harassment: 8`) when reporting CIs / p-values in the paper.
- [ ] **Compile Research Report / Paper Draft:**
  - Integrate final `results/REPORT.md`, `main_benchmark.tex`, `hypothesis_verdicts.tex`, `human_judge_agreement.json`, Logit Lens token examples (`results/tomography/*.json`), and break-even curves into the manuscript (applying external GPU/token pricing only at write-up time on top of raw `cost_table.csv`).
- [ ] **Reproducibility Archive:** Tag final commit hash (`git rev-parse HEAD`) matching `manifest.json` shards across all `results/runs/` and ensure no large model weights or secrets were added to git.
