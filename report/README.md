# MinorSafe Research Report & Manuscript (`report/`)

This directory contains the initial research paper manuscript for **MinorSafe** (*Latent Representation Engineering and Runtime Activation Steering for Age-Appropriate Conversational Systems (Ages 12–18)*).

> **Current Status**: GPU experiments are currently running via the unattended MSBA autopilot pipeline (`scripts/restart_remote.sh` / `scripts/run_autopilot.sh` / `scripts/run_msba.sh`). This manuscript establishes the research motivation, research questions (`RQ-A`, `RQ-B`), pre-registered hypotheses (`H1`–`H10`), methodology (`M0`–`M3`, including token-level hyperplane calibration, dev hyperparameter sweeps, and pure-sociolect orthogonalization), dataset and coined-neologism design, ongoing experiment specifications, and evaluation/statistical protocols. **All empirical result tables, post-rebuild corpus counts, figure panels, hypothesis verdicts, and conclusions are strictly left as `[DATA PENDING]` placeholders until the benchmark completes and `minorsafe analyze` finishes.**

## Directory Contents

| File | Description |
| :--- | :--- |
| [`paper.md`](./paper.md) | Full-length academic manuscript in GitHub-Flavored Markdown (with KaTeX equations, Mermaid architecture diagram, pre-registered hypothesis rules, and structured `[DATA PENDING]` placeholders). |
| [`main.tex`](./main.tex) | Parallel LaTeX manuscript with matching sections, formal equations, and commented `\input` / `\includegraphics` hooks pointing to `../results/tables/` and `../results/figures/`. |
| [`references.bib`](./references.bib) | BibTeX bibliography for the LaTeX manuscript. |
| [`README.md`](./README.md) | This guide explaining directory layout and post-experiment integration workflow. |

## Integrating Empirical Results After Experiments Finish

Once the remote GPU pipeline (`data`, `tomography`, `guards`, `recomposition`, `subculture`, `main`, `rampup`, `quant`, plus `analyze` and human calibration) finishes and results are pulled locally (`bash scripts/remote.sh pull`):

1. **Aggregate benchmark outputs** (if not already completed by `run_msba.sh`):
   ```bash
   minorsafe analyze --runs results/runs --out results
   ```
   This produces:
   - `results/cost_table.csv` (including `n_calib_requested`, `n_calib_effective`, raw token/time costs, and `sri`)
   - `results/REPORT.md`
   - `results/tables/main_benchmark.{md,tex}`
   - `results/tables/hypothesis_verdicts.{md,tex}`
   - `results/figures/fig1_pareto_cost_tokens.{pdf,png}` through `fig8_quantization_scaling.{pdf,png}`
   - Alongside `results/data_card.md`, `results/data_synth_setup.json`, `results/phase_timings.csv`, and `results/tomography/summary.json` (with `auroc_eval_modes`).
2. **Compute human-judge agreement** (after labeling the 150 stratified blinded items from `minorsafe judge export-human --n 150`):
   ```bash
   minorsafe judge agreement --human-csv results/human_eval_150_labeled.csv --out results/human_judge_agreement.json
   ```
3. **Populate manuscript placeholders**:
   - Update Section 4.2 with the regenerated split/category/pair counts from `results/data_card.md` and teacher runtime/tokens from `results/data_synth_setup.json`.
   - Insert observed metrics, 95% clustered bootstrap confidence intervals, Holm-Bonferroni corrected p-values, and raw token/wall-clock cost figures from `results/cost_table.csv`, `results/REPORT.md`, `results/tomography/summary.json`, and `results/human_judge_agreement.json` into Sections 7–9 of [`paper.md`](./paper.md) and [`main.tex`](./main.tex).
   - Replace the placeholder abstract findings and update hypothesis verdicts (`SUPPORTED` / `MIXED` / `NOT_SUPPORTED`) strictly from `results/tables/hypothesis_verdicts.{md,tex}`.
   - Uncomment the `\includegraphics` and `\input` directives in [`main.tex`](./main.tex) and compile with `pdflatex` + `bibtex`.
