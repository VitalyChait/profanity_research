# MinorSafe Repository Audit: Validated Critical & High Issues

**Scope**: Full static and dynamic audit of configs (`configs/`), core library (`src/minorsafe/`), operational scripts (`scripts/`, `setup.sh`, `Makefile`), tests (`tests/`), and run logs/artifacts (`results/`).  
**Baseline offline suite**: `.venv/bin/python -m pytest -q -m "not gpu"` (`119 passed, 3 deselected`) and `.venv/bin/ruff check src tests scripts` pass, meaning these bugs are **not** caught by existing offline CPU tests.  
**Validation policy**: Every issue below was traced in source/configs and dynamically reproduced or empirically verified. Candidates that turned out not to be bugs are documented in [Ruled-Out Candidates](#ruled-out-candidates-false-positives).

---

## Summary Table

| ID | Severity | Area | Primary Location(s) | Validation | Core Impact |
| --- | --- | --- | --- | --- | --- |
| **C1** | **CRITICAL** | Data / Runner / Analysis | `data/obfuscation.py:158`, `runner/run.py:49`, `data/splits.py:75`, `analysis/aggregate.py:215`, `analysis/hypotheses.py:362` | Verified (`0/469` `categories[0]=="obfuscated"`; smoke H6 `NOT_EVALUATED`) | `obfuscated` has 0 rows in `responses.parquet` -> H6 permanently `NOT_EVALUATED`; `label_unsafe=True` obfuscated prompts leak into `benign` and `slang` |
| **C2** | **CRITICAL** | Tomography / Analysis | `methods/steering/tomography.py:234`, `analysis/hypotheses.py:616-618` | Verified (`isinstance(loaded, list)` rejects dict written by tomography) | Real `tomography/summary.json` is silently ignored -> H5 and H7 permanently `NOT_EVALUATED` |
| **C3** | **CRITICAL** | Guards / Experiment Configs | `configs/methods/m1_*.yaml`, `configs/experiments/recomposition.yaml`, `main_grid.yaml`, `methods/guards/method.py:239-279` | Verified (`expand_experiment_grid` + `GuardResponseMethod.respond()` repro) | `placement: both` blocks `profanity`/`obfuscated` at input with hardcoded `fallback`, bypassing `mlm_substitute`, `llm_rewrite`, `regenerate` (invalidates H4) |
| **H1** | **HIGH** | Steering Setup Accounting | `methods/steering/method.py:62-76`, `methods/steering/sweep.py:82-185` | Verified (`SetupReport` returns `1390` tok / `0.039s`, drops sweep `4088` tok / `0.448s`) | Hyperparameter sweep tokens and GPU time are omitted from `setup.json`, biasing H1 (`<1%` of LoRA setup) and TCO curves |
| **H2** | **HIGH** | Caching & Quantized Steering | `methods/steering/sweep.py:82`, `methods/steering/method.py:107-145`, `methods/lora/train.py:126`, `methods/guards/train_encoder.py:89` | Verified (`load_steering_artifact` returns stale `s_l=0.669582` vs `1.238718`) | Cache keys omit `quant`/`concepts`/`pooling`/`seed`; `quant_grid` `vector_source_quant='bf16'` loads 2 models concurrently and skips updating `cd.s_l` |
| **H3** | **HIGH** | Hardware Grid & Multi-GPU | `configs/hardware/gpu_24gb.yaml:39`, `multi_gpu_80gb.yaml:5`, `runner/grid.py:54`, `data/synth.py:138`, `eval/judge.py:147` | Verified (`grep` + `expand_experiment_grid("main_grid", "gpu_24gb")`) | `lora_large_mode: qlora` and `num_gpus: 2` are dead fields -> 7B–9B LoRA runs in `bf16` on 24GB GPUs (OOM); 2-GPU profile runs single-GPU |
| **H4** | **HIGH** | Data / Subculture Metrics | `data/neologisms.py:389-391,596`, `data/obfuscation.py:158`, `analysis/aggregate.py:216-224` | Verified (`237/369` offline and `24/36` seed+obf `slang` rows have `label_unsafe=True`) | `slang_fpr` averages `over_refusal` over ~65% unsafe prompts (where `over_refusal=False`), underestimating benign slang FPR by ~3x in H5 |
| **H5** | **HIGH** | Synthetic Pair Filtering | `data/synth.py:76-91`, `results/data_card.md:34`, `results/logs/data_20261120_002253.log:84` | Verified (`1585` dropped; `mature_theme` collapsed to `44` total / `35` calib) | Detoxify `neg_tox > 0.45` filter deletes ~97% of non-profane `mature_theme` pairs, starving `m3_probe_guard` and `m2_lora` |
| **H6** | **HIGH** | Recomposition References | `data/sources.py:168-220`, `data/obfuscation.py:166`, `eval/semantic.py:180`, `results/data_card.md` | Verified (`toxic_chat*` + `beavertails*` lack `reference_safe`) | ~94% (`1,330/1,422`) of obfuscated/profanity eval rows lack `reference_safe`, causing BERTScore/SARI in `recomposition` to compare responses against unsafe prompts |
| **H7** | **HIGH** | Eval Performance & VRAM | `runner/run.py:337-351`, `eval/evaluate.py:105-159`, `eval/safety.py:20-31`, `eval/judge.py:139-160` | Verified (`182` shards in `main_grid`, `220` in `ramp_up`, etc.) | Per-shard reload of 24B vLLM judge, BERTScorer, and `gpt2-large` wastes hours; `_DETOXIFY_MODELS` stays pinned in CUDA VRAM before vLLM init |
| **H8** | **HIGH** | Reasoning & Judge Length | `generation.py:34-41,166-170`, `eval/judge.py:151`, `configs/prompts.yaml` | Verified (`DeepSeek-R1-Distill-Qwen-1.5B` ends prompt with `<think>\n`) | Truncated DeepSeek-R1 outputs lack `<think>`/`</think>` so `split_reasoning` leaks raw CoT into `answer` with `n_reasoning=0`; `max_model_len=2048` risks vLLM crashes |
| **H9** | **HIGH** | MLM Recomposition Actions | `methods/actions.py:83-97, 119-130` | Verified (`ModernBERT-base` vocab + `"That shitake..."` -> `"That stuffake..."`) | Subword filter only checks `##` (missing ByteLevel `Ġ`/SentencePiece `▁` subpieces); `.find()` lacks word boundaries and mutates innocent words |
| **H10** | **HIGH** | Figures & Paper Tables | `analysis/plots.py:58-131`, `analysis/tables.py:86-109`, `cli.py:290-304` | Verified (code inspection + multi-experiment `cli.analyze` flow) | `fig1`–`fig6` and `main_benchmark.{md,tex}` never filter by `experiment`; `fig3` plots `df.head(6)`, `fig5` overwrites bars across models/experiments |
| **H11** | **HIGH** | Capability Extraction | `eval/capability.py:27, 42-48, 144-150` | Verified (`_extract_mmlu_choice("A careful ... answer is C.") == "A"`) | Leading `^\s*([A-D])\b` matches indefinite article `"A "` / `"C "` before `"the answer is ..."`, corrupting MMLU retention scores (H8) |
| **H12** | **HIGH** | Dead Experiment Pipeline | `data/multiturn.py:42`, `configs/experiments/multiturn.yaml:1-19` | Verified (`grep` shows 0 callers in `src/` or `tests/`) | `build_multiturn_drift_dialogues()` is never wired into dataset building or `multiturn`; the experiment runs on generic single-turn `dev` rows |

---

## Detailed Critical Issues

### C1 — `obfuscated` category absent from `categories[0]`, breaking H6 and polluting `benign` / `slang`
- **Locations**:
  - `src/minorsafe/data/obfuscation.py:153-165` (`generate_obfuscated_examples`)
  - `src/minorsafe/runner/run.py:49, 65` (`_output_to_row`)
  - `src/minorsafe/data/splits.py:75` (`stratified_subsample`)
  - `src/minorsafe/analysis/aggregate.py:215-219` (`aggregate_experiment_results`)
  - `src/minorsafe/analysis/hypotheses.py:362-412` (`_h6`)
- **Root Cause**:
  `generate_obfuscated_examples` creates obfuscated prompts, injects `"What the fuck is this bullshit?"` onto safe prompts lacking profanity (`obfuscation.py:154`), sets `label_unsafe=True` (`line 164`), and builds `cats = list(dict.fromkeys([*ex.categories, "obfuscated", "profanity"]))` (`line 158`). Because `*ex.categories` always comes first (starting with `"benign"`, `"unsafe_other"`, `"slang"`, `"benign_tricky"`, etc.), `"obfuscated"` is never `ex.categories[0]`. However, both `stratified_subsample` (`splits.py:75`) and `_output_to_row` (`runner/run.py:49`) assign `cat = ex.categories[0] if ex.categories else "benign"`. Later, `aggregate.py:215` filters strictly by `df[df["category"] == "obfuscated"]`.
- **Validation**:
  - Running `build_dataset(load_data_spec(), ..., offline=True)` produced `469` examples where `185` contain `"obfuscated"` in `ex.categories`, but **`0` have `ex.categories[0] == "obfuscated"`** (`profanity`: 16, `benign`: 8, `slang`: 117, `benign_tricky`: 16, `harassment`: 4, `violence`: 8, `drugs`: 8, `complex_lexicon`: 8).
  - Running `generate_obfuscated_examples(load_seed_corpus(48), per_sentence=1, seed=13)` produced `48` examples (`label_unsafe=True`): `0` had `categories[0] == "obfuscated"`, and `4` had `categories[0] == "benign"` with `label_unsafe=True`.
  - In the production build metadata (`results/data_card.md:24, 44-63`), `obfuscated: 1422` appears only because the data card sums all tags inside `categories`; `1,330` of those rows come from `beavertails_obf_*` (`236`) and `toxic_chat_obf_*` (`1,094`), whose `categories[0]` is `"benign"` or `"unsafe_other"`.
  - Running `run_smoke_pipeline(..., max_examples=12)` yielded `category` counts `{'slang': 20, 'benign_tricky': 20, 'drugs': 10, 'profanity': 10}` (`obfuscated` = `0`), all `obfuscated_success` / `lexicon_obfuscated_recall` = `NaN`, and `H6: status=NOT_EVALUATED`.
- **Impact**: Hypothesis **H6** cannot be evaluated (`NOT_EVALUATED`), all `obfuscated_*` metrics are `NaN`, and `label_unsafe=True` obfuscated profanity contaminates `"benign"` and `"benign_tricky"` evaluation buckets.
- **Recommended Fix**: Place `"obfuscated"` first (`["obfuscated", "profanity", ...]`) in `generate_obfuscated_examples` (or preserve a `categories: list[str]` column alongside primary category and match tag membership in `aggregate.py:215`), and ensure benign source prompts injected with unsafe profanity are no longer tagged `"benign"`.

---

### C2 — `tomography/summary.json` dict-vs-list schema mismatch breaks H5 and H7 (`NOT_EVALUATED`)
- **Locations**:
  - `src/minorsafe/methods/steering/tomography.py:234` (`run_tomography_for_profile`)
  - `src/minorsafe/analysis/hypotheses.py:316-360, 415-469, 612-618` (`_h5`, `_h7`, `evaluate_hypotheses`)
- **Root Cause**:
  `run_tomography_for_profile` writes `summary.json` as a dictionary:
  `write_json(out_root / "summary.json", {"profile": resolved, "models": summaries})` (`tomography.py:234`).
  Conversely, `evaluate_hypotheses` reads `summary.json` and checks:
  `if isinstance(loaded, list): tomo_summary = loaded` (`hypotheses.py:617`), silently ignoring `dict` objects and leaving `tomo_summary = None`.
  *(Note: `tests/unit/test_path_a.py` passed by writing a bare `list` fixture rather than invoking `run_tomography_for_profile`.)*
- **Validation**:
  Passing a dict formatted identically to `tomography.py:234` (`{"profile": "cpu_debug", "models": [...]}`) to `evaluate_hypotheses(...)` on `run_smoke_pipeline` output returned:
  - `H5: status=NOT_EVALUATED | evidence=Missing tomography summary (run \`minorsafe tomography\` to produce results/tomography/summary.json).`
  - `H7: status=NOT_EVALUATED | evidence=Missing tomography summary (run \`minorsafe tomography\` to produce results/tomography/summary.json).`
- **Impact**: Real pipeline runs of `minorsafe tomography` followed by `minorsafe analyze` can never evaluate **H5** or **H7**.
- **Recommended Fix**: Accept both `list` and `loaded.get("models")` in `evaluate_hypotheses` (`hypotheses.py:616-618`) and add a unit test validating the actual `run_tomography_for_profile` output against `evaluate_hypotheses`.

---

### C3 — `placement: both` default on `M1` guards blocks `profanity`/`obfuscated` prompts at input, bypassing `mlm_substitute`, `llm_rewrite`, and `regenerate`
- **Locations**:
  - `configs/methods/m1_encoder_guard.yaml:10-11`, `configs/methods/m1_decoder_guard.yaml:6-7`
  - `configs/experiments/recomposition.yaml:16-30` (`m1_enc_mlm_sub`, `m1_enc_llm_rewrite`, `m1_dec_llm_rewrite`)
  - `configs/experiments/main_grid.yaml:15-22`, `configs/experiments/subculture_audit.yaml:22-31`
  - `src/minorsafe/methods/guards/method.py:239-279` (`GuardResponseMethod.respond`)
  - `src/minorsafe/analysis/hypotheses.py:277-313` (`_h4`)
- **Root Cause**:
  Both `m1_encoder_guard.yaml` and `m1_decoder_guard.yaml` default to `placement: both`, and neither `recomposition.yaml`, `main_grid.yaml`, nor `subculture_audit.yaml` overrides `placement: output` when testing post-generation actions (`mlm_substitute`, `llm_rewrite`, `regenerate`). In `GuardResponseMethod.respond()`, if `self.placement in ("input", "both")` and `in_sc >= self.threshold`, the method immediately assigns `blocked=True`, `action_taken="fallback"`, and returns a canned refusal (`guards/method.py:257-277`) without adding the item to `need_gen` — completely skipping draft generation and `execute_moderation_action`.
- **Validation**:
  - Expanding `recomposition`, `main_grid`, and `subculture_audit` via `expand_experiment_grid` confirms every `m1_encoder_guard` and `m1_decoder_guard` arm (`mlm_substitute`, `llm_rewrite`, `regenerate`) has `placement="both"`.
  - Instantiating `GuardResponseMethod` with `action="mlm_substitute"` (recomposition config) and calling `.respond()` on `seed_corpus` `profanity`/`obfuscated` examples returned `blocked=True action_taken='fallback' in_sc=0.880` with text `"I can't help with that specific request..."` even on safe prompts (`label_unsafe=False`).
- **Impact**:
  - **Invalidates H4 (`recomposition`)**: `m1_enc_mlm_sub`, `m1_enc_llm_rewrite`, and `m1_dec_llm_rewrite` return canned fallbacks (near-0 generation/rewrite tokens and latency) on flagged `profanity`/`obfuscated` inputs instead of executing MLM substitution or LLM rewriting.
  - **Collapses `main_grid` action sweeps**: `regenerate`, `llm_rewrite`, and `mlm_substitute` arms behave identically to `fallback` whenever the input guard fires.
- **Recommended Fix**: Set `placement: output` in `recomposition.yaml`, `subculture_audit.yaml` (for recomposition/rewrite arms), and `main_grid.yaml` action sweeps when evaluating `regenerate`/`llm_rewrite`/`mlm_substitute` (or reserve `placement: both` solely for `action: fallback`), and validate in `GuardResponseMethod` when non-fallback actions are paired with input-blocking placement.

---

## Detailed High Issues

### H1 — Steering hyperparameter sweep cost (`wall_s`, `gpu_s`, `tokens`) is discarded from `setup.json`, inflating H1
- **Locations**: `src/minorsafe/methods/steering/method.py:62-76`, `src/minorsafe/methods/steering/sweep.py:82-185`, `configs/methods/m3_steer_sttp.yaml:10` (and sibling `m3_steer_*.yaml` configs), `src/minorsafe/analysis/hypotheses.py:153-190` (`_h1`)
- **Root Cause**: `SteeringResponseMethod.setup()` invokes `swept_best, _ = sweep_steering_hyperparams(...)` (`method.py:63`), throwing away the returned `SetupReport` (`_`). Only the final single-layer `extract_or_load_directions` `SetupReport` is returned and written to `setup.json`, even though `sweep_steering_hyperparams` evaluates ~18–24 `(depth, alpha/offset)` candidates over `max_dev=16` generations plus multi-layer activation captures.
- **Validation**: Running `SteeringResponseMethod.setup()` with `force_sweep=True`, `n_calib=10`, and 6 dev examples (`task-338`) produced:
  - Sweep cache `sweep_qwen25_05b_sttp_pca_N10.json`: `total_tokens=4088`, `wall_s=0.4475`.
  - Returned `SetupReport`: single stage `extract_vectors` with `total_tokens=1390`, `wall_s=0.0391`.
- **Impact**: Systematically under-reports `m3_steer_*` `setup_gpu_s`, `setup_wall_s`, and `setup_tokens`, distorting pre-registered hypothesis **H1** (`setup_gpu_s(StTP) < 0.01 * LoRA`) and TCO break-even comparisons.
- **Recommended Fix**: Merge the sweep `SetupReport` stage(s) into the returned `SetupReport` in `SteeringResponseMethod.setup()` (preserving `cached` / stage provenance).

---

### H2 — Artifact cache key collisions across `quant`/`concepts`/`pooling`/`seed`, plus stale `cd.s_l` and double-model VRAM allocation on `vector_source_quant='bf16'` (`quant_grid` / H10)
- **Locations**:
  - `src/minorsafe/methods/steering/sweep.py:82` (`sweep_steering_hyperparams`)
  - `src/minorsafe/methods/steering/method.py:48, 107-145` (`SteeringResponseMethod.setup`)
  - `src/minorsafe/methods/lora/train.py:126` (`train_lora_adapter`)
  - `src/minorsafe/methods/guards/train_encoder.py:89` (`train_encoder_guard`)
  - `configs/experiments/quant_grid.yaml:17-21`, `configs/experiments/ramp_up.yaml`
- **Root Cause**:
  1. `sweep.py:82` sets `setup_key = f"sweep_{gen_spec.alias}_{mode}_{estimator}_N{n_calib}"`, omitting `quant`, `concepts`, `pooling`, and `orthogonal_concept`. In `quant_grid.yaml`, `bf16` writes the sweep JSON and subsequent `int8` / `nf4` runs reuse it instead of sweeping on the quantized model.
  2. In `steering/method.py:107-145` (`vector_source_quant == "bf16"` and `ctx.quant != "bf16"`):
     - `self._init_generator(ctx)` (`line 48`) keeps the quantized model in GPU VRAM while `load_decoder(..., quant="bf16")` (`line 122`) simultaneously loads a second full BF16 model.
     - On cache hit at `load_steering_artifact(ctx.artifact_dir, bf16_key)` (`line 118`), `method.py` skips the `cd.s_l` recalculation that `extract_or_load_directions` runs (`extract.py:546-549`) for the current/swept `target_offset_sigma`, and `line 144` unconditionally calls `save_steering_artifact` overwriting `steering_meta.json` on disk with `rep.cached = True`.
  3. `lora/train.py:126` (`lora_{alias}_{quant}_N{n_calib}_r{r}_ep{epochs}`) omits `concepts`, `lr`, `lora_alpha`, `batch_size`, and `seed`; `train_encoder.py:89` (`enc_{alias}_N{n_calib}_ep{epochs}_lr{lr:g}`) omits `quant` and `seed`.
- **Validation**: Cache-hit test (`task-338`) extracted directions at `target_offset_sigma=0.5` (`s_l=0.669582`). Re-loading with `target_offset_sigma=1.0` via `extract_or_load_directions` updated `s_l=1.238718`, whereas `load_steering_artifact` (used at `steering/method.py:118`) returned stale `s_l=0.669582`.
- **Impact**: Corrupted target offset `s_l` and leaked `bf16` sweep hyperparameters in `quant_grid` (**H10**), avoidable peak VRAM spikes (`int8/nf4` + `bf16` resident simultaneously), and silent cross-configuration cache collisions.
- **Recommended Fix**: Include all distinguishing parameters (`quant`, hashed `concepts`, `pooling`, `orthogonal_concept`, `lr`, `batch_size`, `seed`) in all artifact `setup_key`s; extract BF16 directions before loading the target quantized generator (or release/skip holding both concurrently); recompute `cd.s_l` on cache hit in `steering/method.py:118-120` and save only on cache miss.

---

### H3 — `lora_large_mode` and `num_gpus` are unimplemented dead config fields (BF16 LoRA OOMs on 24GB GPUs; multi-GPU profile stays single-GPU)
- **Locations**: `configs/hardware/gpu_24gb.yaml:39`, `gpu_48gb.yaml:43`, `multi_gpu_80gb.yaml:5, 41`, `src/minorsafe/runner/grid.py:54`, `src/minorsafe/data/synth.py:138`, `src/minorsafe/eval/judge.py:147`, `src/minorsafe/telemetry/cost.py:36`
- **Root Cause**:
  `grep` over `src/` shows `lora_large_mode` (`gpu_24gb.yaml:39`, `gpu_48gb.yaml:43`) is never referenced outside `config.py:78`. `expand_experiment_grid` (`grid.py:54`) always picks `profile.default_quant` (`bf16`) regardless of model size (`params_b`) or `lora_large_mode == "qlora"`. Similarly, `HardwareProfile.num_gpus` (`multi_gpu_80gb.yaml:5` = `2`) is only checked in `hardware.py:626` and is never passed to `SetupReport(num_gpus=...)` (`cost.py:36`), `vllm.LLM(..., tensor_parallel_size=...)` (`synth.py:138`, `judge.py:147`), or multi-GPU shard execution.
- **Validation**:
  Expanding `main_grid` on `gpu_24gb` (`lora_large_mode: qlora`) assigned `m2_lora` with `quant='bf16'` to all 13 models, including 7B–9B models (`qwen25_7b`, `deepseek_r1_8b`, `gemma2_9b`, `llama31_8b`). On `multi_gpu_80gb` (`num_gpus: 2`, `teacher: llama33_70b_awq`), `vllm.LLM` defaults to `tensor_parallel_size=1` on GPU 0 and `aggregate.py:249` falls back to `num_gpus=1` when multiplying `gpu_s_per_query`.
- **Impact**: `gpu_24gb` cannot train 7B–9B LoRA adapters without OOM; `multi_gpu_80gb` fails to utilize multiple GPUs and misreports multi-GPU `gpu_s`.
- **Recommended Fix**: Honor `profile.lora_large_mode == "qlora"` in `expand_experiment_grid` (e.g., use `nf4` for `m2_lora` when `size_group == "L"` or `params_b >= 6.0`), and wire `profile.num_gpus` into `SetupReport` and vLLM `tensor_parallel_size`.

---

### H4 — `slang_fpr` (`H5`) is diluted (~2.8x–3x) by `label_unsafe=True` prompts inside `category == "slang"`
- **Locations**: `src/minorsafe/data/neologisms.py:389-391, 596`, `src/minorsafe/data/obfuscation.py:158, 164`, `src/minorsafe/analysis/aggregate.py:216, 220-224`, `src/minorsafe/eval/evaluate.py:172`, `src/minorsafe/analysis/hypotheses.py:343-351`
- **Root Cause**:
  `aggregate.py:216-224` calculates `slang_fpr = slang_df["over_refusal"].mean()` over `df[df["category"] == "slang"]`. Because `over_refusal = bool(is_refusal and not label_unsafe)` (`evaluate.py:172`), any row with `label_unsafe=True` always has `over_refusal=False`. However:
  1. `neologisms.py:389-391, 596` puts `"slang"` at `categories[0]` for both benign and toxic neologisms (`label_unsafe=True`).
  2. `obfuscation.py:158, 164` preserves `categories[0] == "slang"` while marking obfuscated benign slang `label_unsafe=True`.
- **Validation**:
  - In `build_dataset(load_data_spec(), ..., offline=True)`, **`237` of `369` (`64.2%`)** examples with `categories[0] == "slang"` had `label_unsafe=True`.
  - In `load_seed_corpus(48) + generate_obfuscated_examples(..., per_sentence=2, seed=13)` (reproducing `results/data_card.md:26`'s exact `slang: 36` count), **`24` of `36` (`66.7%`)** had `label_unsafe=True`, leaving only `12` benign slang rows.
- **Impact**: `slang_fpr` does not measure false-positive rate on benign sociolect — it is diluted ~2.8x–3.0x toward 0, compressing effect sizes in **H5**.
- **Recommended Fix**: Restrict `slang_fpr` (`aggregate.py:216`) to benign slang rows (`(df["category"] == "slang") & (~df["label_unsafe"].astype(bool))`), and stop assigning `categories[0] == "slang"` to toxic or obfuscated `label_unsafe=True` items.

---

### H5 — Detoxify filter wipes out ~97% of `mature_theme` contrastive pairs (`44` total / `35` calib), starving `m3_probe_guard` and `m2_lora`
- **Locations**: `src/minorsafe/data/synth.py:76-91` (`filter_contrastive_pairs`), `configs/methods/m3_probe_guard.yaml`, `configs/methods/m2_lora.yaml`, `results/data_card.md:34`, `results/logs/data_20261120_002253.log:80-84`
- **Root Cause**:
  `filter_contrastive_pairs` (`synth.py:76-91`) groups `mature_theme` with `profanity` and `slang_toxic` and rejects every pair whose `negative` completion lacks profanity tokens and has Detoxify `toxicity_score <= 0.45` (`dropped_negative_weak`). Because `mature_theme` explicitly targets non-profane age-inappropriate/harmful content (dangerous dares, choking challenges, mature guidance without swear words), Detoxify (trained on profane/toxic comments) scores nearly all negative completions `<= 0.45`.
- **Validation**:
  Verified in `results/logs/data_20261120_002253.log:80-84` (`dropped_negative_weak: 1585` out of `8750` input pairs) and `results/data_card.md:31-37`: `mature_theme` collapsed to **`44` total pairs (`35` calib, `4` dev, `5` test)**, vs `1,145` `profanity`, `1,160` `lexical_complexity`, and `875` `slang_benign`.
- **Impact**: `mature_theme` drops to <4% of `m2_lora`/`m3_probe_guard` training pairs (`configs/methods/m3_probe_guard.yaml:10-13`), creating severe concept imbalance and early saturation across `ramp_up`.
- **Recommended Fix**: Exclude `mature_theme` from the Detoxify `<= 0.45` profanity/toxicity rejection threshold (or validate `mature_theme` pairs with a semantic/judge check rather than profane toxicity).

---

### H6 — Most `recomposition` (`profanity`, `obfuscated`) examples lack `reference_safe`, causing BERTScore/SARI to score safe replies against unsafe prompts
- **Locations**: `src/minorsafe/data/sources.py:168-220`, `src/minorsafe/data/obfuscation.py:166`, `src/minorsafe/eval/semantic.py:180`, `configs/experiments/recomposition.yaml:4`, `results/data_card.md:40-64`
- **Root Cause**:
  `load_toxic_chat` (`sources.py:176-192`) and `load_beavertails` (`sources.py:203-220`) leave `reference_safe=None`, and `generate_obfuscated_examples` copies `reference_safe=ex.reference_safe` (`obfuscation.py:166`). In `compute_recomposition_metrics_batch` (`semantic.py:180`), missing `reference_safe` falls back to `clean_srcs` (the raw prompt).
- **Validation**:
  In `results/data_card.md:40-64`, `toxic_chat_obf_*` (`1,094`) and `beavertails_obf_*` (`236`) account for `1,330` of the `1,422` obfuscated/profanity examples lacking `reference_safe`, all selected by `recomposition.yaml` (`categories: ["profanity", "complex_lexicon", "obfuscated"]`) and falling back to the raw profane user prompt in `semantic.py:180`.
- **Impact**: `bertscore_f1_ref`, `sari_score`, and `recompose_filter_acc` (`bs_ref >= 0.70`) reward echoing an unsafe/obfuscated prompt rather than an age-appropriate reference on those subsets in **H4** (`recomposition`).
- **Recommended Fix**: Either populate `reference_safe` for `toxic_chat`/`beavertails` (or benign recomposition subsets) or use `baseline_m0` / judge-evaluated meaning preservation cleanly instead of grading safe answers against unsafe prompts via `bertscore_f1_ref`.

---

### H7 — Per-shard reload of 24B vLLM judge, BERTScorer, and `gpt2-large` (104–220 times per experiment) + pinned `_DETOXIFY_MODELS` on CUDA
- **Locations**: `src/minorsafe/runner/run.py:337-351`, `src/minorsafe/eval/evaluate.py:105-159`, `src/minorsafe/eval/judge.py:139-160, 279-296`, `src/minorsafe/eval/safety.py:20-31`, `configs/hardware/gpu_24gb.yaml:41, 47`
- **Root Cause**:
  `run_experiment` calls `evaluate_responses_df(...)` inside `for ... in ordered_eval:` (`runner/run.py:337-351`), running **182** times in `main_grid`, **220** in `ramp_up`, **143** in `quant_grid`, **104** in `recomposition`, and **104** in `subculture_audit`. Every invocation constructs and destroys `BERTScorer` (`roberta-large`), `gpt2-large`, and `LLMJudge` (`mistral_small_24b_awq` via vLLM `LLM(...)`). Meanwhile `_get_detoxify` (`safety.py:20-31`) keeps `Detoxify("original")` permanently in module-level `_DETOXIFY_MODELS` on CUDA without freeing it, leaving it resident when vLLM attempts to allocate `vllm_gpu_memory_utilization` (`0.85`–`0.88`).
- **Validation**: Direct code trace + grid expansion counts (`182` shards in `main_grid`, `220` in `ramp_up`, etc.); `results/logs/data_20261120_002253.log:68` shows a single vLLM engine init taking `290.28s` for AWQ model setup + CUDA graphs, while `gpu_24gb.yaml:41,47` pairs 24B AWQ (`~14 GB` weights) with `gpu_memory_utilization=0.85` alongside resident Detoxify.
- **Impact**: Massive cold-start overhead across ~780 total shards and elevated OOM / free-VRAM assertion risk during vLLM judge startup on 24GB GPUs.
- **Recommended Fix**: Reuse `LLMJudge`, `BERTScorer`, `gpt2-large`, and `Detoxify` across shards within a single `run_experiment` / `cmd_eval` eval phase (sequentially per stage: Detoxify -> free -> BERTScorer -> free -> GPT-2 -> free -> batched vLLM Judge across all shards), explicitly releasing `_DETOXIFY_MODELS` before vLLM init.

---

### H8 — DeepSeek-R1 `split_reasoning()` misclassifies truncated CoT as visible answer + `LLMJudge` hardcoded `max_model_len=2048` crash risk
- **Locations**: `src/minorsafe/generation.py:34-41, 166-170`, `src/minorsafe/eval/judge.py:151`, `configs/prompts.yaml`
- **Root Cause**:
  1. DeepSeek-R1-Distill chat templates append `<｜Assistant｜><think>\n` at the end of the prompt. Generated completions (`ids[:n]`) therefore omit `<think>` and only emit `</think>` once reasoning finishes. When generation truncates at `max_new_tokens=1024` inside `<think>`, `raw` contains neither `<think>` nor `</think>`, so `split_reasoning(raw)` returns `reasoning=""`, `answer=raw.strip()`, `unterminated=False` (`n_reasoning=0`).
  2. `LLMJudge._ensure_loaded()` hardcodes `max_model_len=2048` (`judge.py:151`). With `judge_rubric` (1,457 chars / 221 words), multi-turn/long prompts, and up to 1,024-token truncated CoT responses in `text`, prompt lengths can exceed 2,048 tokens and raise an uncaught vLLM `ValueError`.
- **Validation**:
  Verified directly against `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` tokenizer: `format_chat` output ends with `'<｜Assistant｜><think>\n'`, and `split_reasoning("Okay, let's think carefully...")` returned `reasoning='', unterminated=False` (`n_reasoning=0`).
- **Impact**: Unclosed DeepSeek-R1 chain-of-thought leaks into safety/toxicity/BERTScore/judge evaluation and undercounts `tok_generator_reasoning_out` in **H9**, while long judge inputs risk crashing eval.
- **Recommended Fix**: Detect prompt-ending `<think>` in `ChatGenerator.generate` (`format_chat(...)` ends with `<think>`) so unclosed generations are marked `unterminated=True` with `n_reasoning=n` and `text=""`, and guard/truncate oversized judge inputs or raise `max_model_len` cleanly in `LLMJudge`.

---

### H9 — `MLMRecomposer` fails to filter ModernBERT/XLM-R continuation subwords and corrupts unrelated words via unbounded `.find()`
- **Locations**: `src/minorsafe/methods/actions.py:83-97, 119-130` (`MLMRecomposer._predict_replacement`, `MLMRecomposer.recompose`)
- **Root Cause**:
  1. `_predict_replacement` (`actions.py:86`) skips subword continuations using only `raw_piece.startswith("##")` (WordPiece). All configured MLM models in `configs/models.yaml` (`modernbert_base_mlm`, `modernbert_large_mlm`, `xlmr_large_mlm`) use ByteLevel (`Ġ` word-initial) or SentencePiece (`▁` word-initial) vocabularies where continuation subwords have **no** `##` prefix.
  2. `recompose` (`actions.py:120`) searches targets via `cur.lower().find(orig.lower())` without word boundaries, replacing matching substrings inside benign words earlier in the string.
- **Validation**:
  - Verified `answerdotai/ModernBERT-base` tokenizer contains continuation tokens `'ing'`, `'ed'`, `'ly'`, `'tion'`, `'s'`, `'ment'` without `##` while whole words carry `'Ġ'`.
  - Direct `MLMRecomposer.recompose` execution produced:
    - `'That shitake soup tasted like shit.'` -> `'That stuffake soup tasted like that.'`
    - `'The pissing contest started with piss.'` -> `'The annoying contest started with that.'`
    - `'The badger was a really bad pet.'` (`extra_suspect_words=['bad']`) -> `'The thatger was a really bad pet.'` (leaving `"bad"` unredacted).
- **Impact**: Breaks fluency and meaning preservation while potentially leaving actual profanity/suspect spans intact in `mlm_substitute` (**H4**).
- **Recommended Fix**: Enforce word-start markers (`Ġ`/`▁` or offset checks) for BPE/SentencePiece MLMs, and replace using span offsets / `\b...\b` regex boundaries rather than unbounded `.find()`.

---

### H10 — `analysis/plots.py` and `tables.py` never filter by `experiment`, garbling paper figures (`fig1`–`fig6`) and `main_benchmark.tex`
- **Locations**: `src/minorsafe/analysis/plots.py:58-131`, `src/minorsafe/analysis/tables.py:86-109`, `src/minorsafe/cli.py:290-304`
- **Root Cause**:
  `cmd_analyze` (`cli.py:296-301`) aggregates all experiments under `results/runs/` (`main_grid`, `guards`, `recomposition`, `subculture_audit`, `ramp_up`, `quant_grid`, etc.) into one `summary_df` and passes it unfiltered to `generate_all_figures` and `generate_all_tables`:
  - `fig3_tco_breakeven_curves` (`plots.py:81`) iterates `df.head(6)` — arbitrary first 6 rows in Parquet traversal order (e.g., 6 `guards` runs) instead of one curve per method family on `main_grid`.
  - `fig5_encoder_vs_decoder_guards` (`plots.py:117`) plots `ax.bar(df["method"].astype(str), df["e2e_ms_p50"])` across all rows/experiments/models, overwriting duplicate `method` bars.
  - `fig1`, `fig2`, `fig4`, `fig6`, and `main_benchmark.{md,tex}` mix rows from incompatible splits (`dev` vs `test`) and category subsets (`recomposition` vs `main_grid`).
- **Impact**: Paper-ready figures and `main_benchmark.tex` generated by `minorsafe analyze` are scientifically invalid whenever multiple experiment directories exist.
- **Recommended Fix**: Filter `fig1`/`fig2`/`fig3` and `main_benchmark` to `experiment == "main_grid"`, `fig4` to `ramp_up`, `fig5` to `guards` (`m1_*`/`m3_probe`), and `fig6` to `recomposition`, aggregating across models/seeds cleanly before plotting.

---

### H11 — `_extract_mmlu_choice` regex matches leading `"A "` / `"C "` before `"the answer is X"`, corrupting MMLU scores (H8)
- **Locations**: `src/minorsafe/eval/capability.py:27, 42-48, 144-150`
- **Root Cause**:
  `_LETTER_RE = re.compile(r"(?:answer(?:\s+is)?[:\s]*|^\s*|\()([A-D])(?:\)|\b)", re.IGNORECASE)` uses `search()`, which returns the leftmost match. Because `^\s*([A-D])\b` matches at index 0, any conversational reply starting with the article `"A "` or `"C "` (or `"C programming..."`) matches `'A'`/`'C'` immediately rather than a later `"answer is C"`.
- **Validation**:
  Direct execution of `_extract_mmlu_choice`:
  - `"A careful calculation shows the answer is C."` -> `'A'`
  - `"C programming is fast. Answer: B"` -> `'C'`
- **Impact**: Systematically misgrades verbose / CoT MMLU answers, corrupting `mmlu_acc` and `mmlu_delta` in **H8**.
- **Recommended Fix**: Check explicit `"answer is X"` / `"option X"` / `"\\boxed{X}"` patterns first (taking the last explicit match), falling back to standalone leading/trailing letters only when no explicit answer phrase exists.

---

### H12 — `multiturn` experiment is disconnected from `build_multiturn_drift_dialogues()` (dead code)
- **Locations**: `src/minorsafe/data/multiturn.py:42`, `configs/experiments/multiturn.yaml:1-19`
- **Root Cause**: `grep -rnE "build_multiturn_drift_dialogues"` across `src/`, `configs/`, `scripts/`, and `tests/` shows zero callers outside its definition in `multiturn.py:42`. Neither `data/sources.py`, `data/build.py`, nor `runner/run.py` loads multi-turn dialogues, and `multiturn.yaml` has no custom source/category filter.
- **Impact**: Running `minorsafe run -e multiturn` evaluates 60 single-turn `dev` examples from `examples.parquet` rather than conversational drift dialogues.
- **Recommended Fix**: Register `build_multiturn_drift_dialogues` as a source loader in `data/sources.py` (or `runner/run.py`) tagged with a dedicated category/source and filter `configs/experiments/multiturn.yaml` to it.

---

## Ruled-Out Candidates (False Positives)

The following potential issues were explicitly tested during validation and confirmed **not** to be bugs:
1. **`tinyBenchmarks/tinyMMLU` and `tinyBenchmarks/tinyGSM8K` loading (`eval/capability.py`)**: Verified via `datasets.load_dataset(..., split="test", streaming=True)` that both load cleanly with `test` splits and the exact column names (`question`, `choices`, `answer`) expected by `capability.py`.
2. **`configs/models.yaml` HuggingFace repository IDs**: Verified via `HfApi().model_info(...)` that all configured `hf_id` and `quant_ids` entries resolve on HuggingFace Hub.
3. **`bert_score==0.3.13` + `roberta-large` compatibility with `transformers==5.22.2` (`eval/semantic.py`)**: Verified tokenizer (`use_fast=False`), `bert_score.utils.sent_encode`, and `RobertaModel.encoder.layer` execute cleanly in the repo's `.venv`.
4. **`run_autopilot.sh` default `PROFILE=gpu_lab1`**: Verified `configs/hardware/gpu_lab1.yaml` exists in the repository (`43` lines).
5. **Hardware profiles referencing gated `gemma2_27b_it` (`hardware.py`)**: Checked all `configs/hardware/*.yaml` profiles (`gpu_24gb`, `gpu_48gb`, `gpu_80gb`, `gpu_gb10`, `gpu_lab1`, `multi_gpu_80gb`); all configure `eval.judge_model: mistral_small_24b_awq` (`gated: false`), not `gemma2_27b_it`.
6. **Prefill hook alignment in `SteeringResponseMethod._generate_batch_with_prefill_probe` (`steering/method.py:174-195`)**: Left-padding places the last prompt token at index `-1` during prefill step 0, and `if not captured:` ignores subsequent 1-token decode steps cleanly.
