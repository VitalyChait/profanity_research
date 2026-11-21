#!/usr/bin/env python3
"""Pre-flight HuggingFace model downloader and cache verifier for a hardware profile.

Models go into the hub cache that ``from_pretrained`` and vLLM read (``$HF_HUB_CACHE``, by default
``$HF_HOME/hub``), and only the files they load are fetched: configs, tokenizers, custom modeling code and
one copy of the weights. Skipped are Meta's ``original/`` checkpoints, Mistral-native
``consolidated.safetensors`` duplicates, and ``.bin`` weights when safetensors exist.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from minorsafe.config import load_data_spec, load_hardware, load_models
from minorsafe.hardware import resolve_profile_name

# Non-weight files transformers / vLLM may read (config, generation config, tokenizer, chat template, code).
META_PATTERNS = ("*.json", "*.txt", "*.py", "*.model", "*.tiktoken", "*.jinja", "tokenizer*")
IGNORE_PATTERNS = ("original/*",)
# Used only when the repo listing shows no root-level weights we recognise.
FALLBACK_WEIGHT_PATTERNS = ("*.safetensors", "*.bin")


def collect_profile_models(profile_name: str) -> list[tuple[str, str, bool]]:
    models = load_models()
    hw = load_hardware(profile_name)
    aliases: list[str] = []
    for group in (hw.generators, hw.decoder_guards, hw.encoder_guards, hw.encoder_mlms):
        for a in group:
            if a not in aliases and a in models:
                aliases.append(a)
    if hw.teacher and hw.teacher in models and hw.teacher not in aliases:
        aliases.append(hw.teacher)
    if profile_name != "cpu_debug":
        data_teacher = load_data_spec().synth_teacher
        if data_teacher and data_teacher in models and data_teacher not in aliases:
            aliases.append(data_teacher)
    if hw.eval.judge_model and hw.eval.judge_model in models and hw.eval.judge_model not in aliases:
        aliases.append(hw.eval.judge_model)

    out = []
    for a in aliases:
        m = models[a]
        out.append((a, m.hf_id, m.gated))
    return out


def weight_files(repo_files: list[str]) -> list[str]:
    """One loadable copy of the weights: root-level HF safetensors, else root-level PyTorch ``.bin`` shards."""
    root = [f for f in repo_files if "/" not in f and not f.startswith("consolidated")]
    safetensors = sorted(f for f in root if f.endswith(".safetensors"))
    if safetensors:
        return safetensors
    return sorted(
        f for f in root if f.endswith(".bin") and (f.startswith("pytorch_model") or f.startswith("model"))
    )


def download_patterns(repo_files: list[str]) -> list[str]:
    weights = weight_files(repo_files)
    return [*META_PATTERNS, *(weights or FALLBACK_WEIGHT_PATTERNS)]


def runtime_hub_cache() -> Path:
    """The hub cache ``from_pretrained`` and vLLM resolve models from."""
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def stray_hub_downloads(hf_home: str | Path | None, hub_cache: Path) -> list[Path]:
    """Model repos that older versions of this script downloaded into ``$HF_HOME`` instead of its hub cache."""
    if not hf_home:
        return []
    home = Path(hf_home)
    if not home.is_dir() or home.resolve() == hub_cache.resolve():
        return []
    return sorted(p for p in home.glob("models--*") if p.is_dir())


def _dir_size_gb(path: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            if not fp.is_symlink():
                total += fp.stat().st_size
    return total / 1e9


def _load_config_and_tokenizer(hf_id: str, cache_dir: str | None) -> None:
    """Resolve config + tokenizer from the local cache only, as the runtime would (custom-code fallback too)."""
    from transformers import AutoConfig, AutoTokenizer

    kw = dict(cache_dir=cache_dir, local_files_only=True)
    try:
        AutoConfig.from_pretrained(hf_id, **kw)
    except (ValueError, KeyError):
        # e.g. EuroBERT ships custom modeling code (same fallback as registry.load_encoder_classifier)
        AutoConfig.from_pretrained(hf_id, trust_remote_code=True, **kw)
    AutoTokenizer.from_pretrained(hf_id, **kw)


def download_models(
    entries: list[tuple[str, str, bool]], cache_dir: str | None = None, verify: bool = False
) -> int:
    """Download each repo's runtime files into ``cache_dir`` (None = the runtime hub cache)."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    failures = 0
    for alias, hf_id, gated in entries:
        if gated and not os.environ.get("HF_TOKEN"):
            print(f"WARNING: {alias} ({hf_id}) is gated and HF_TOKEN is not set.", file=sys.stderr)
        repo_files = api.list_repo_files(hf_id)
        weights = weight_files(repo_files)
        if not weights:
            print(
                f"WARNING: {alias} ({hf_id}): no root-level weights recognised; downloading all "
                f"{'/'.join(FALLBACK_WEIGHT_PATTERNS)} files.",
                file=sys.stderr,
            )
        print(f"Downloading {alias} -> {hf_id} ({len(weights)} weight file(s)) ...")
        local = Path(
            snapshot_download(
                repo_id=hf_id,
                cache_dir=cache_dir,
                allow_patterns=download_patterns(repo_files),
                ignore_patterns=list(IGNORE_PATTERNS),
            )
        )
        if verify:
            missing = [w for w in weights if not (local / w).exists()]
            if missing:
                print(f"[FAIL] {alias} ({hf_id}): missing weight files {missing}", file=sys.stderr)
                failures += 1
                continue
            _load_config_and_tokenizer(hf_id, cache_dir)
            print(f"Verified {alias} ({hf_id}) in {local}")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download or verify HuggingFace models for a hardware profile."
    )
    parser.add_argument(
        "--profile", default="cpu_debug", help="Hardware profile name in configs/hardware/, or 'auto'"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print required models without downloading")
    parser.add_argument("--verify", action="store_true", help="Verify tokenizer/config load for each model")
    parser.add_argument(
        "--check-gates",
        action="store_true",
        help="Verify HuggingFace read authorization for all gated models without downloading weights",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HF *hub* cache to download into (default: $HF_HUB_CACHE, i.e. $HF_HOME/hub, which is where "
        "from_pretrained and vLLM look). Not $HF_HOME itself.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    profile = resolve_profile_name(args.profile)
    entries = collect_profile_models(profile)
    print(f"Profile '{profile}' requires {len(entries)} model(s):")
    for alias, hf_id, gated in entries:
        gate_tag = " [GATED]" if gated else ""
        print(f"  - {alias}: {hf_id}{gate_tag}")

    if args.dry_run:
        return 0

    if args.check_gates:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN") or None
        api = HfApi(token=token)
        gated_entries = [(a, h) for a, h, g in entries if g]
        if not gated_entries:
            print(f"Profile '{profile}' has no gated models.")
            return 0
        failures: list[tuple[str, str, str]] = []
        for alias, hf_id in gated_entries:
            try:
                api.auth_check(repo_id=hf_id, token=token)
                print(f"[PASS] Gate check: {alias} ({hf_id})")
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}".splitlines()[0]
                print(f"[FAIL] Gate check: {alias} ({hf_id}) -> {msg}", file=sys.stderr)
                failures.append((alias, hf_id, msg))
        if failures:
            print(
                "\nAction required: accept the license terms on HuggingFace and ensure HF_TOKEN is set for:",
                file=sys.stderr,
            )
            for alias, hf_id, _ in failures:
                print(f"  - {alias}: https://huggingface.co/{hf_id}", file=sys.stderr)
            return 1
        print(f"All {len(gated_entries)} gated model(s) authorized for profile '{profile}'.")
        return 0

    hub_cache = runtime_hub_cache()
    print(f"Hub cache: {args.cache_dir or hub_cache}")
    if args.cache_dir and Path(args.cache_dir).resolve() != hub_cache.resolve():
        print(
            f"WARNING: --cache-dir {args.cache_dir} is not the runtime hub cache ({hub_cache}); "
            "from_pretrained/vLLM will not find these downloads unless HF_HUB_CACHE points there.",
            file=sys.stderr,
        )
    stray = stray_hub_downloads(os.environ.get("HF_HOME"), hub_cache)
    if stray:
        size = sum(_dir_size_gb(p) for p in stray)
        print(
            f"WARNING: {len(stray)} model repo(s) ({size:.1f} GB) sit directly in $HF_HOME, outside the "
            f"hub cache, so nothing reads them (left by an earlier download_models.py). Reclaim the disk with:"
            f"\n  rm -rf " + " ".join(str(p) for p in stray),
            file=sys.stderr,
        )

    return download_models(entries, cache_dir=args.cache_dir, verify=args.verify)


if __name__ == "__main__":
    raise SystemExit(main())
