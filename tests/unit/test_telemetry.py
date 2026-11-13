"""Phase 2 unit tests: Stopwatch, TimingLogitsProcessor, TokenUsage, SetupReport, cost & break-even."""

from __future__ import annotations

import math
import time

import pytest

from minorsafe.config import ModelSpec
from minorsafe.generation import ChatGenerator, split_reasoning
from minorsafe.telemetry.cost import SetupReport, break_even, efficiency, total_cost
from minorsafe.telemetry.gpu import GPUSampler
from minorsafe.telemetry.timer import Stopwatch
from minorsafe.telemetry.tokens import TokenUsage

pytestmark = pytest.mark.unit


def test_stopwatch_nested_spans():
    sw = Stopwatch(sync_cuda=False)
    with sw.span("outer"):
        time.sleep(0.01)
        with sw.span("inner"):
            time.sleep(0.01)
    assert sw.get("inner") >= 0.008
    assert sw.get("outer") >= sw.get("inner")
    assert sw.counts["outer"] == 1
    assert sw.counts["inner"] == 1
    assert sw.total(["inner"]) == pytest.approx(sw.get("inner"))


def test_token_usage_accounting_and_addition():
    u1 = TokenUsage(generator_in=50, generator_out=100, generator_reasoning_out=40, guard_in=50, guard_out=4)
    u2 = TokenUsage(rewrite_in=120, rewrite_out=90, encoder_in=60)
    combined = u1 + u2
    assert combined.input_total == 50 + 50 + 120 + 60
    assert combined.output_total == 100 + 40 + 4 + 90
    assert combined.total == combined.input_total + combined.output_total
    d = combined.to_dict()
    assert d["tok_total"] == combined.total
    assert d["tok_generator_reasoning_out"] == 40


def test_setup_report_and_cost_breakeven():
    rep = SetupReport(method="m3_steer_sttp", setup_key="test_key", num_gpus=1)
    with rep.stage("extract_vectors", examples=100) as st:
        st.tokens = 12500
        time.sleep(0.005)
    assert rep.wall_s > 0.0
    assert rep.tokens == 12500
    roundtrip = SetupReport.from_dict(rep.to_dict())
    assert roundtrip.method == "m3_steer_sttp"
    assert roundtrip.tokens == 12500

    # Break-even math:
    # Method A (Steering): setup = 120s, per_query = 1.0s
    # Method B (Decoder Guard): setup = 0s, per_query = 1.6s
    # Crossover N = (120 - 0) / (1.6 - 1.0) = 200 queries
    n_cross = break_even(setup_a=120.0, per_query_a=1.0, setup_b=0.0, per_query_b=1.6)
    assert n_cross == pytest.approx(200.0)
    assert total_cost(120.0, 1.0, 200) == pytest.approx(total_cost(0.0, 1.6, 200))

    # If A is cheaper in both setup and per-query, break-even is 0
    assert break_even(0.0, 0.5, 100.0, 1.0) == 0.0
    # If A has higher setup AND higher per-query cost, break-even is inf
    assert math.isinf(break_even(100.0, 2.0, 0.0, 1.0))

    eff = efficiency(success_rate=0.90, tokens_per_query=150.0, seconds_per_query=1.5)
    assert eff["success_per_1k_tokens"] == pytest.approx(6.0)
    assert eff["tokens_per_success"] == pytest.approx(150.0 / 0.90)


def test_split_reasoning_and_chat_generator_telemetry(dummy_lm, dummy_tokenizer):
    reason, ans, unterm = split_reasoning("<think>let me calculate 2+2</think>The answer is 4.")
    assert reason == "let me calculate 2+2"
    assert ans == "The answer is 4."
    assert unterm is False

    _, ans2, unterm2 = split_reasoning("<think>unterminated reasoning chain")
    assert ans2 == ""
    assert unterm2 is True

    spec = ModelSpec(
        alias="deepseek_dummy",
        hf_id="dummy",
        arch="decoder",
        role="generator",
        family="deepseek",
        size_group="M",
        reasoning=True,
        chat_template=False,
    )
    dummy_tokenizer.emit_reasoning = True
    gen = ChatGenerator(dummy_lm, dummy_tokenizer, spec=spec, max_new_tokens=6, min_new_tokens=4)
    res = gen.generate([[{"role": "user", "content": "Hello world 😀"}]], seed=0)
    assert len(res) == 1
    r = res[0]
    assert r.n_in >= 1
    assert r.n_reasoning > 0
    assert r.n_out >= 1
    assert "<think>" not in r.text
    assert 0.0 <= r.ttft_s <= r.e2e_s
    assert r.decode_tok_per_s > 0.0


def test_gpu_sampler_cpu_safe():
    with GPUSampler(interval_s=0.01) as s:
        time.sleep(0.02)
    assert s.energy_j >= 0.0
    assert s.peak_mem_mb >= 0.0
