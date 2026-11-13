#!/usr/bin/env python3
"""Pre-flight HuggingFace model downloader and cache verifier for a hardware profile."""

from __future__ import annotations

import argparse
import os
import sys

from minorsafe.config import load_hardware, load_models


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
    if hw.eval.judge_model and hw.eval.judge_model in models and hw.eval.judge_model not in aliases:
        aliases.append(hw.eval.judge_model)

    out = []
    for a in aliases:
        m = models[a]
        out.append((a, m.hf_id, m.gated))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download or verify HuggingFace models for a hardware profile."
    )
    parser.add_argument("--profile", default="cpu_debug", help="Hardware profile name in configs/hardware/")
    parser.add_argument("--dry-run", action="store_true", help="Print required models without downloading")
    parser.add_argument("--verify", action="store_true", help="Verify tokenizer/config load for each model")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"), help="Optional HF cache dir")
    args = parser.parse_args()

    entries = collect_profile_models(args.profile)
    print(f"Profile '{args.profile}' requires {len(entries)} model(s):")
    for alias, hf_id, gated in entries:
        gate_tag = " [GATED]" if gated else ""
        print(f"  - {alias}: {hf_id}{gate_tag}")

    if args.dry_run:
        return 0

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    for alias, hf_id, gated in entries:
        if gated and not os.environ.get("HF_TOKEN"):
            print(f"WARNING: {alias} ({hf_id}) is gated and HF_TOKEN is not set.", file=sys.stderr)
        print(f"Downloading {alias} -> {hf_id} ...")
        snapshot_download(repo_id=hf_id, cache_dir=args.cache_dir)
        if args.verify:
            AutoConfig.from_pretrained(hf_id, cache_dir=args.cache_dir)
            AutoTokenizer.from_pretrained(hf_id, cache_dir=args.cache_dir)
            print(f"Verified {alias} ({hf_id})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
