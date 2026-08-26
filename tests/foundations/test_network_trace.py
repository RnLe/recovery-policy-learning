"""Network trace export: schema, determinism, and agreement with the rollout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gr_foundations import network_trace


@pytest.fixture(scope="module")
def trace():
    root = Path.cwd()
    if not (root / "data" / "foundations" / network_trace.CHECKPOINT).exists():
        pytest.skip("lab04 checkpoint not present; run grf run lab04 first")
    return network_trace.build_trace(root)


def test_schema_and_shapes(trace) -> None:
    assert trace["kind"] == "network-flow" and trace["variant"] == "full"
    assert trace["exported_steps"] == len(trace["steps"])
    assert trace["weights"]["word_embedding"]["shape"] == [
        len(trace["mission"]["vocab"]), 32,
    ]
    assert trace["weights"]["head"]["shape"] == [3, network_trace.HEAD_COLUMNS]
    step = trace["steps"][0]
    acts = step["acts"]
    assert acts["obs_vec"]["shape"] == [64]
    assert acts["fused"]["shape"] == [128] and acts["hidden"]["shape"] == [128]
    assert acts["logits"]["shape"] == [3] and acts["probs"]["shape"] == [3]
    assert acts["conv1_sample"]["shape"] == [len(network_trace.CONV_OUT_CHANNELS), 5, 5]
    assert acts["conv2_sample"]["shape"] == [len(network_trace.CONV_OUT_CHANNELS), 3, 3]
    assert step["prev_action"] is None  # the start token precedes any action
    assert trace["mission_acts"]["per_token_hidden"]["shape"] == [
        len(trace["mission"]["tokens"]), 64,
    ]


def test_blocks_carry_sane_ranges(trace) -> None:
    for name, block in trace["weights"].items():
        tensor = block["tensor"] if "tensor" in block else block
        assert tensor["min"] <= tensor["max"], name
        expected = 1
        for dim in tensor["shape"]:
            expected *= dim
        assert len(tensor["values"]) == expected, name
    for key, bounds in trace["ranges"].items():
        assert bounds["min"] <= bounds["max"], key


def test_logits_agree_with_the_recorded_rollout(trace) -> None:
    for step in trace["steps"]:
        values = step["acts"]["logits"]["values"]
        assert values.index(max(values)) == step["action"]
        probs = step["acts"]["probs"]["values"]
        assert abs(sum(probs) - 1.0) < 1e-3


def test_export_is_deterministic(trace) -> None:
    again = network_trace.build_trace(Path.cwd())
    a = json.dumps(trace, sort_keys=True)
    b = json.dumps(again, sort_keys=True)
    assert a == b
