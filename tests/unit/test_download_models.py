"""claude_issues_found.md H8: pre-downloads must land in the hub cache the runtime reads, without duplicate weights."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from minorsafe.utils import REPO_ROOT

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def dm():
    spec = importlib.util.spec_from_file_location(
        "download_models", REPO_ROOT / "scripts" / "download_models.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_cache_dir_is_the_runtime_hub_cache(dm):
    # None -> huggingface_hub uses HF_HUB_CACHE ($HF_HOME/hub), where from_pretrained and vLLM look.
    assert dm.build_parser().parse_args([]).cache_dir is None


def test_weight_selection_keeps_one_runtime_copy(dm):
    llama = [
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "original/consolidated.00.pth",
        "original/params.json",
        "tokenizer.json",
    ]
    assert dm.weight_files(llama) == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    # Mistral-native duplicate and PyTorch/Flax/ONNX copies are not runtime weights.
    mixed = [
        "consolidated.safetensors",
        "model.safetensors",
        "pytorch_model.bin",
        "flax_model.msgpack",
        "onnx/model.onnx",
    ]
    assert dm.weight_files(mixed) == ["model.safetensors"]
    # .bin-only repos (s-nlp/roberta_toxicity_classifier, facebook/xlm-roberta-xxl) still get their weights.
    assert dm.weight_files(["config.json", "pytorch_model.bin", "training_args.bin"]) == ["pytorch_model.bin"]
    # Custom modeling code (EuroBERT) and tokenizer files are always fetched.
    pats = dm.download_patterns(["modeling_eurobert.py", "model.safetensors"])
    assert {"*.py", "*.json", "tokenizer*", "model.safetensors"} <= set(pats)
    assert "*.bin" not in pats
    assert dm.download_patterns(["weights/x.safetensors"])[-2:] == ["*.safetensors", "*.bin"]


def test_download_uses_runtime_cache_and_filters(dm, monkeypatch, tmp_path: Path):
    calls = []

    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id):
            return ["config.json", "model.safetensors", "pytorch_model.bin", "original/consolidated.00.pth"]

    def fake_snapshot(**kw):
        calls.append(kw)
        out = tmp_path / kw["repo_id"].replace("/", "--")
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.safetensors").write_bytes(b"x")
        return str(out)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    monkeypatch.setattr(dm, "_load_config_and_tokenizer", lambda hf_id, cache_dir: None)

    assert dm.download_models([("m", "org/m", False)], verify=True) == 0
    (kw,) = calls
    assert kw["cache_dir"] is None
    assert "model.safetensors" in kw["allow_patterns"]
    assert not any(p.endswith(".bin") for p in kw["allow_patterns"])
    assert "original/*" in kw["ignore_patterns"]


def test_verify_fails_when_weights_are_missing(dm, monkeypatch, tmp_path: Path):
    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id):
            return ["config.json", "model.safetensors"]

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: str(tmp_path))
    monkeypatch.setattr(dm, "_load_config_and_tokenizer", lambda hf_id, cache_dir: None)
    assert dm.download_models([("m", "org/m", False)], verify=True) == 1


def test_stray_downloads_in_hf_home_are_reported(dm, tmp_path: Path):
    hf_home = tmp_path / "hfhome"
    (hf_home / "hub" / "models--Qwen--Qwen2.5-7B-Instruct").mkdir(parents=True)
    (hf_home / "models--Qwen--Qwen2.5-7B-Instruct").mkdir()
    assert dm.stray_hub_downloads(hf_home, hf_home / "hub") == [hf_home / "models--Qwen--Qwen2.5-7B-Instruct"]
    # HF_HUB_CACHE == HF_HOME (an unusual but valid setup): those repos are the live cache.
    assert dm.stray_hub_downloads(hf_home, hf_home) == []
    assert dm.stray_hub_downloads(None, hf_home / "hub") == []
