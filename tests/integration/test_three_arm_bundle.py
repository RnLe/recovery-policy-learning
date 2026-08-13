"""Tiny three-arm vertical slice (gate G5 semantics).

One complete miniature pipeline bundle: shared base data and checkpoint,
bit-identical arm clones, exact per-round budgets in both augmented arms,
equal optimizer updates and target exposures, an unchanged base checkpoint,
and full determinism of the whole pipeline under the seed bundle.
"""

from __future__ import annotations

import pytest
import torch

from grounded_recovery.artifacts import read_jsonl
from grounded_recovery.config import contract_hash, load_and_validate
from grounded_recovery.data import make_manifests
from grounded_recovery.experiment import ARM_EXTRA, ARM_RECOVERY, run_pilot_bundle

TINY = {
    "data.n0": 20,
    "data.b": 6,
    "data.k": 2,
    "data.h": 2,
    "perturbation.collection_time_set": [1, 2, 3],
    "training.base_updates": 60,
    "training.base_targets_per_update": 8,
    "training.new_targets_per_update": 4,
    "training.updates_per_round": 15,
    "data.split_counts.collection": 14,
}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, tiny_config_factory):
    cfg = load_and_validate(tiny_config_factory(**TINY))
    manifest_root = tmp_path_factory.mktemp("manifests")
    make_manifests(cfg, manifest_root)
    data_root = tmp_path_factory.mktemp("data")
    summary = run_pilot_bundle(cfg, "B00", manifest_root, data_root)
    return cfg, manifest_root, data_root, summary


def test_exact_budgets_in_both_arms(bundle) -> None:
    _cfg, _mroot, _droot, summary = bundle
    for arm in (ARM_EXTRA, ARM_RECOVERY):
        collections = summary["arms"][arm]["collections"]
        assert [c["round_index"] for c in collections] == [1, 2]
        assert all(c["budget"] == 3 for c in collections)
        assert all(c["revealed_targets"] == 3 for c in collections)


def test_recovery_labels_are_windowed_and_partial_windows_recorded(bundle) -> None:
    cfg, _mroot, droot, summary = bundle
    # b_k=3 with h=2 forces at least one partial window per round.
    for collection in summary["arms"][ARM_RECOVERY]["collections"]:
        rows = read_jsonl(f"{collection['dataset_dir']}/episode_index.jsonl")
        per_episode = [row["revealed_targets"] for row in rows]
        assert sum(per_episode) == 3
        assert all(count <= cfg.data.h for count in per_episode)
        assert collection["delivered"] + collection["undelivered"] == collection["episodes"]


def test_equal_updates_and_exposures(bundle) -> None:
    _cfg, _mroot, _droot, summary = bundle
    extra = summary["arms"][ARM_EXTRA]["cumulative"]
    recovery = summary["arms"][ARM_RECOVERY]["cumulative"]
    assert extra == recovery
    assert extra["updates"] == 30  # 2 rounds x 15 updates
    assert extra["base"] == 30 * 8
    assert extra["new"] == 30 * 4
    assert summary["fairness"][0]["updates"] == 30


def test_exposure_ledgers_verify_and_match(bundle) -> None:
    cfg, _mroot, droot, summary = bundle
    root = f"{droot}/{summary['contract_hash'][:12]}/B00/training"
    rows = {
        arm: read_jsonl(f"{root}/{arm}/exposure_ledger.jsonl")
        for arm in (ARM_EXTRA, ARM_RECOVERY)
    }
    for arm, arm_rows in rows.items():
        assert len(arm_rows) == 30
        assert all(row["arm"] == arm for row in arm_rows)
        assert all(row["base_targets_drawn"] == 8 for row in arm_rows)
        assert all(row["new_targets_drawn"] == 4 for row in arm_rows)
    # The two arms trained on different data: their parameters must diverge.
    assert (
        summary["arms"][ARM_EXTRA]["final_state_digest"]
        != summary["arms"][ARM_RECOVERY]["final_state_digest"]
    )
    # And each diverged from the shared base.
    for arm in (ARM_EXTRA, ARM_RECOVERY):
        assert summary["arms"][arm]["final_state_digest"] != summary["base"]["state_digest"]


def test_base_checkpoint_untouched_and_final_checkpoints_load(bundle) -> None:
    cfg, _mroot, _droot, summary = bundle
    from grounded_recovery.train import model_state_digest

    payload = torch.load(
        summary["base"]["checkpoint"], map_location="cpu", weights_only=False
    )
    assert model_state_digest(payload["model_state"]) == summary["base"]["state_digest"]
    assert payload["meta"]["arm"] == "base"
    for arm in (ARM_EXTRA, ARM_RECOVERY):
        arm_payload = torch.load(
            summary["arms"][arm]["final_checkpoint"], map_location="cpu",
            weights_only=False,
        )
        assert arm_payload["meta"]["arm"] == arm
        assert arm_payload["meta"]["round_index"] == cfg.data.k
        assert arm_payload["meta"]["update_index"] == 30
        assert arm_payload["meta"]["contract_hash"] == contract_hash(cfg)


def test_recovery_episodes_store_proposals_and_corruption(bundle) -> None:
    import numpy as np

    from grounded_recovery.data import read_episode
    from grounded_recovery.schemas import NULL_ACTION

    cfg, _mroot, _droot, summary = bundle
    collection = summary["arms"][ARM_RECOVERY]["collections"][0]
    rows = read_jsonl(f"{collection['dataset_dir']}/episode_index.jsonl")
    checked_delivered = False
    for row in rows:
        arrays, sidecar = read_episode(
            f"{collection['dataset_dir']}/episodes", row["episode_id"]
        )
        # The learner's proposal is stored at every step of a policy rollout.
        assert not np.any(arrays.policy_proposed_action == NULL_ACTION)
        intervention = sidecar.intervention
        assert intervention is not None
        if intervention["delivered"]:
            t_star = intervention["scheduled_time"]
            mapping = cfg.perturbation.collection_operator.mapping
            proposal = int(arrays.policy_proposed_action[t_star])
            assert int(arrays.executed_action[t_star]) == mapping[proposal]
            assert bool(arrays.perturbation_scheduled[t_star])
            assert bool(arrays.perturbation_delivered[t_star])
            # Off the scheduled step the policy's proposal was executed.
            others = [t for t in range(arrays.length) if t != t_star]
            assert all(
                int(arrays.executed_action[t]) == int(arrays.policy_proposed_action[t])
                for t in others
            )
            revealed_at = np.flatnonzero(arrays.target_revealed)
            if intervention["revealed_targets"] == 0:
                # A corruption on the terminal transition leaves no
                # post-corruption state: zero labels is valid, honest data.
                assert revealed_at.size == 0
                assert arrays.length == t_star + 1
            else:
                # Revealed targets lie strictly after the corrupted transition.
                checked_delivered = True
                assert revealed_at.min() == t_star + 1
    assert checked_delivered


def test_bundle_is_deterministic(bundle, tmp_path_factory) -> None:
    cfg, manifest_root, _droot, summary = bundle
    rerun_root = tmp_path_factory.mktemp("data-rerun")
    rerun = run_pilot_bundle(cfg, "B00", manifest_root, rerun_root)
    assert rerun["base"]["state_digest"] == summary["base"]["state_digest"]
    for arm in (ARM_EXTRA, ARM_RECOVERY):
        assert (
            rerun["arms"][arm]["final_state_digest"]
            == summary["arms"][arm]["final_state_digest"]
        )
        # Collections must agree in every field except the run-local paths.
        for first, second in zip(
            summary["arms"][arm]["collections"],
            rerun["arms"][arm]["collections"],
            strict=True,
        ):
            stripped_first = {k: v for k, v in first.items() if k != "dataset_dir"}
            stripped_second = {k: v for k, v in second.items() if k != "dataset_dir"}
            assert stripped_first == stripped_second


def test_bundle_refuses_unresolved_pilot_fields(bundle, tiny_config_factory, tmp_path) -> None:
    from grounded_recovery.data import ManifestError

    _cfg, manifest_root, _droot, _summary = bundle
    unresolved = load_and_validate(
        tiny_config_factory(**{"training.updates_per_round": "PILOT_TO_FREEZE"})
    )
    with pytest.raises(ManifestError, match="unresolved"):
        run_pilot_bundle(unresolved, "B00", manifest_root, tmp_path / "d")


def test_crossed_evaluation_is_itt_and_paired(bundle) -> None:
    """All three arms on the identical validation panel across the three slices."""
    from grounded_recovery.data import (
        base_dataset_dir,
        load_split_manifest,
        vocabulary_from_dataset,
    )
    from grounded_recovery.evaluate import (
        SLICE_CLEAN,
        SLICE_MATCHED,
        SLICE_UNSEEN,
        evaluate_policy_on_entries,
        load_arm_policy,
    )
    from grounded_recovery.perturbations import operator_from_config
    from grounded_recovery.statistics import paired_difference, success_summary
    from grounded_recovery.world import WorldSession

    cfg, manifest_root, droot, summary = bundle
    entries, _ = load_split_manifest(manifest_root, "validation")
    dataset_dir = base_dataset_dir(cfg, "B00", droot)
    vocab = vocabulary_from_dataset(dataset_dir)

    checkpoints = {
        "bc_base": summary["base"]["checkpoint"],
        ARM_EXTRA: summary["arms"][ARM_EXTRA]["final_checkpoint"],
        ARM_RECOVERY: summary["arms"][ARM_RECOVERY]["final_checkpoint"],
    }
    matched_operator = operator_from_config(
        cfg.perturbation.collection_operator, cfg.environment.action_ids
    )
    unseen_operator = operator_from_config(
        cfg.perturbation.unseen_operator, cfg.environment.action_ids
    )
    unseen_times = (2, 3)  # pilot-mutable; resolved here for the miniature panel
    slices = {
        SLICE_CLEAN: (None, None),
        SLICE_MATCHED: (matched_operator, cfg.perturbation.collection_time_set),
        SLICE_UNSEEN: (unseen_operator, unseen_times),
    }

    session = WorldSession(cfg.environment)
    all_rows: dict[tuple[str, str], list] = {}
    try:
        for arm, checkpoint in checkpoints.items():
            policy = load_arm_policy(cfg, checkpoint, vocab)
            for slice_name, (operator, time_set) in slices.items():
                all_rows[(arm, slice_name)] = evaluate_policy_on_entries(
                    cfg, session, policy, vocab, entries,
                    bundle_id="B00", arm=arm, slice_name=slice_name,
                    operator=operator, time_set=time_set,
                )
    finally:
        session.close()

    for (_arm, slice_name), rows in all_rows.items():
        # Intention-to-treat: exactly one row per assigned scenario, always.
        assert len(rows) == len(entries)
        if slice_name == SLICE_CLEAN:
            assert all(row.scheduled_time is None for row in rows)
            assert all(not row.delivered for row in rows)
        else:
            assert all(row.scheduled_time is not None for row in rows)
        summary_stats = success_summary(rows)
        assert summary_stats.assigned == len(entries)
        assert 0.0 <= summary_stats.rate <= 1.0
        assert summary_stats.delivered <= summary_stats.assigned

    # The crossed panel is identical across arms within a slice, so the paired
    # within-bundle contrast is well defined.
    for slice_name in slices:
        delta = paired_difference(
            all_rows[(ARM_RECOVERY, slice_name)], all_rows[(ARM_EXTRA, slice_name)]
        )
        assert -1.0 <= delta <= 1.0
    # Operators are labeled correctly per slice.
    assert all(r.operator_name == "rot_plus" for r in all_rows[(ARM_EXTRA, SLICE_MATCHED)])
    assert all(r.operator_name == "rot_minus" for r in all_rows[(ARM_EXTRA, SLICE_UNSEEN)])


def test_validation_pilot_report(tmp_path_factory, tiny_config_factory) -> None:
    """`run_validation_pilot` produces the crossed validation report and rows."""
    from grounded_recovery.artifacts import read_json, read_jsonl
    from grounded_recovery.data import make_manifests
    from grounded_recovery.experiment import run_validation_pilot

    overrides = dict(TINY)
    overrides["perturbation.unseen_time_set"] = [2, 3]
    cfg = load_and_validate(tiny_config_factory(**overrides))
    manifest_root = tmp_path_factory.mktemp("manifests-pilot")
    make_manifests(cfg, manifest_root)
    data_root = tmp_path_factory.mktemp("data-pilot")
    report = run_validation_pilot(cfg, "B00", manifest_root, data_root)

    panel = report["panel_scenarios"]
    assert panel == cfg.data.split_counts.validation
    for arm in ("bc_base", ARM_EXTRA, ARM_RECOVERY):
        for slice_name in ("clean", "matched", "unseen"):
            cell = report["success"][arm][slice_name]
            assert cell["assigned"] == panel  # intention-to-treat, always
            assert 0 <= cell["successes"] <= panel
    assert set(report["paired_recovery_minus_extra"]) == {"clean", "matched", "unseen"}

    bundle_root = f"{data_root}/{report['bundle']['contract_hash'][:12]}/B00"
    stored = read_json(f"{bundle_root}/pilot_report.json")
    assert stored["phase"] == "validation_pilot"
    rows = read_jsonl(f"{bundle_root}/validation_evaluation_rows.jsonl")
    assert len(rows) == 3 * 3 * panel  # arms x slices x panel
