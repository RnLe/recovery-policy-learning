"""Lab 7: simulator statistical properties, chain/hash demos, reanalysis."""

from __future__ import annotations

import numpy as np
import pytest

from gr_foundations import lab07_methodology as lab


def test_delivery_bias_simulation_shows_the_designed_bias() -> None:
    result = lab.simulate_delivery_bias(n_scenarios=400, n_replicates=600)
    # The ITT estimator centers on the true ITT effect...
    assert abs(result["itt_bias"]) < 0.01
    # ...while per-protocol is biased by an amount comparable to the effects
    # under study.
    assert abs(result["per_protocol_bias"]) > 0.02
    # The efficient arm delivers less often, which is the mechanism of the bias.
    assert result["delivery_rate_b"] < result["delivery_rate_a"]


def test_delivery_bias_simulation_deterministic() -> None:
    first = lab.simulate_delivery_bias(n_scenarios=100, n_replicates=50)
    second = lab.simulate_delivery_bias(n_scenarios=100, n_replicates=50)
    assert first["itt_mean"] == second["itt_mean"]
    assert np.array_equal(first["per_protocol_estimates"], second["per_protocol_estimates"])


def test_pairing_simulation_prefers_paired_analysis() -> None:
    result = lab.simulate_pairing(n_replicates=400)
    assert result["paired_mean_width"] < result["unpaired_mean_width"]
    assert result["paired_power"] > result["unpaired_power"]
    assert result["paired_power"] > 0.8


def test_hash_chain_demo_catches_the_edit_at_its_row() -> None:
    result = lab.hash_chain_demo()
    assert result["first_mismatch"] == result["tampered_row"]
    assert result["final_hash_honest"] != result["final_hash_tampered"]


def test_contract_hash_demo_changes_prefix(tmp_path) -> None:
    import shutil

    (tmp_path / "configs").mkdir()
    shutil.copyfile(
        "configs/experiment_contract.yaml",
        tmp_path / "configs" / "experiment_contract.yaml",
    )
    result = lab.contract_hash_demo(tmp_path)
    assert result["original_prefix"] != result["edited_prefix"]


def test_reanalysis_arithmetic() -> None:
    rows = []
    for arm, successes in (("recovery", [1, 1, 1, 0]), ("extra", [1, 0, 0, 0])):
        for index, success in enumerate(successes):
            rows.append(
                {
                    "arm": arm,
                    "slice": "unseen",
                    "delivered": index < 3,  # last row undelivered
                    "success": bool(success),
                }
            )
    result = lab.reanalyze_lab06(rows)
    assert result["itt_delta"] == pytest.approx(0.75 - 0.25)
    assert result["per_protocol_delta"] == pytest.approx(1.0 - 1 / 3)
    assert result["itt_denominators"] == [4, 4]
    assert result["per_protocol_denominators"] == [3, 3]


def test_run_requires_lab06(tmp_path) -> None:
    import shutil

    (tmp_path / "configs").mkdir()
    shutil.copyfile(
        "configs/experiment_contract.yaml",
        tmp_path / "configs" / "experiment_contract.yaml",
    )
    paths = lab.LabPaths(lab_id="lab07", repo_root=tmp_path)
    with pytest.raises(lab.FoundationsError, match="lab06"):
        lab.run(paths, force=False)


def test_run_tiny_on_lab06_outputs(tmp_path) -> None:
    import json
    import shutil

    from gr_foundations import lab06_shift

    (tmp_path / "configs").mkdir()
    shutil.copyfile(
        "configs/experiment_contract.yaml",
        tmp_path / "configs" / "experiment_contract.yaml",
    )
    lab06_shift.run(
        lab.LabPaths(lab_id="lab06", repo_root=tmp_path),
        force=False,
        base_episodes=6,
        holdout_scenarios=4,
        budget_labels=10,
        window=3,
        base_updates=12,
        arm_updates=8,
        n_seeds=1,
        sweep_times=(2,),
        sweep_scenarios=2,
    )
    paths = lab.LabPaths(lab_id="lab07", repo_root=tmp_path)
    summary = lab.run(paths, force=False, skip_unmatched=True, sim_replicates=60)
    metrics = json.loads((paths.out_dir / "metrics.json").read_text())
    assert metrics["metrics"]["unmatched_arm"] is None
    assert "lab06_reanalysis" in metrics["metrics"]
    for name in ("itt_bias.svg", "paired_power.svg"):
        assert (paths.figures_dir / name).exists()
    assert (paths.report_dir / "freeze_mechanisms.typ").exists()
    assert summary["unmatched_unseen"] == "skipped"
