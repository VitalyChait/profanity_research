# Optional reproducible CUDA container image for MinorSafe (when Docker GPU runtime is permitted).
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONNOUSERSITE=1 \
    PIP_CONFIG_FILE=/dev/null \
    UV_NO_MODIFY_PATH=1 \
    TOKENIZERS_PARALLELISM=false \
    UV_PROJECT_ENVIRONMENT=/workspace/.venv \
    UV_CACHE_DIR=/workspace/.cache/uv \
    HF_HOME=/workspace/.cache/huggingface \
    PATH=/workspace/.venv/bin:/root/.local/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY configs ./configs
COPY src ./src
COPY scripts ./scripts

RUN uv sync --python 3.11 --reinstall-package minorsafe --extra gpu --extra vllm --extra dev

ENTRYPOINT ["/workspace/.venv/bin/python", "-m", "minorsafe.cli"]
CMD ["--help"]
