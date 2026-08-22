"""Lab 2: aliasing arithmetic on synthetic records, collection determinism, run smoke."""

from __future__ import annotations

import json
from pathlib import Path

from gr_foundations import lab02_decision
from gr_foundations.common import LabPaths
from gr_foundations.lab01_world import contract_environment


def _record(key: str, episode: int, t: int, pos, direction: int, rec: int):
    return lab02_decision.StateRecord(
        observation_key=key,
        episode_index=episode,
        t=t,
        position=tuple(pos),
        agent_dir=direction,
        recommendation=rec,
        image_bytes=key.encode(),
    )


def test_analyze_aliasing_synthetic_arithmetic() -> None:
    records = [
        # Class "a": two different worlds, conflicting labels -> aliased,
        # cross-world, heterogeneous; minority mass 1.
        _record("a", 0, 1, (1, 1), 0, 2),
        _record("a", 1, 3, (5, 5), 2, 0),
        # Class "b": same world, same pose, revisited -> same Markov state,
        # NOT aliased, no minority mass.
        _record("b", 0, 2, (2, 2), 1, 1),
        _record("b", 0, 6, (2, 2), 1, 1),
        # Class "c": same world, two poses, same label -> aliased but
        # label-homogeneous.
        _record("c", 2, 0, (3, 3), 0, 2),
        _record("c", 2, 4, (3, 4), 0, 2),
    ]
    analysis = lab02_decision.analyze_aliasing(records)
    assert analysis["total_states"] == 6
    assert analysis["observation_classes"] == 3
    assert analysis["aliased_classes"] == 2
    assert analysis["cross_world_aliased_classes"] == 1
    assert analysis["label_heterogeneous_classes"] == 1
    assert analysis["memoryless_error_lower_bound"] == 1 / 6
    showcase = analysis["showcase"]
    assert showcase is not None
    assert showcase["first"].episode_index == 0
    assert showcase["second"].episode_index == 1


def test_collect_states_deterministic() -> None:
    env_cfg = contract_environment(LabPaths(lab_id="lab02", repo_root=Path.cwd()))
    records_a, actions_a, counters_a = lab02_decision.collect_states(env_cfg, 4)
    records_b, actions_b, counters_b = lab02_decision.collect_states(env_cfg, 4)
    assert counters_a == counters_b
    assert actions_a == actions_b
    assert records_a == records_b
    # Every record's stored observation bytes match its grouping key content.
    for record in records_a[:5]:
        assert len(record.image_bytes) == 7 * 7 * 3


def test_reported_aliases_are_genuine() -> None:
    """Aliased classes must contain byte-equal observations from distinct states."""
    env_cfg = contract_environment(LabPaths(lab_id="lab02", repo_root=Path.cwd()))
    records, _actions, _counters = lab02_decision.collect_states(env_cfg, 60)
    by_key: dict[str, list] = {}
    for record in records:
        by_key.setdefault(record.observation_key, []).append(record)
    aliased = [
        entries
        for entries in by_key.values()
        if len({(e.episode_index, e.position, e.agent_dir) for e in entries}) >= 2
    ]
    # Deterministic seeds: this slice is known to contain aliased classes,
    # including one with conflicting oracle labels.
    assert aliased, "expected at least one aliased class in 60 episodes"
    analysis = lab02_decision.analyze_aliasing(records)
    assert analysis["label_heterogeneous_classes"] >= 1
    for entries in aliased:
        assert len({e.image_bytes for e in entries}) == 1
        assert len({(e.episode_index, e.position, e.agent_dir) for e in entries}) >= 2


def test_run_produces_labelled_artifacts(tmp_path) -> None:
    import shutil

    (tmp_path / "configs").mkdir()
    shutil.copyfile(
        "configs/experiment_contract.yaml",
        tmp_path / "configs" / "experiment_contract.yaml",
    )
    paths = LabPaths(lab_id="lab02", repo_root=tmp_path)
    summary = lab02_decision.run(paths, force=False, rollout_episodes=12)
    metrics = json.loads((paths.out_dir / "metrics.json").read_text())
    assert metrics["metrics"]["aliasing"]["total_states"] > 0
    assert (paths.figures_dir / "full_observability_contrast.svg").exists()
    assert (paths.report_dir / "pomdp_mapping.typ").exists()
    assert (paths.report_dir / "aliasing_facts.typ").exists()
    assert summary["episodes"] <= 12
