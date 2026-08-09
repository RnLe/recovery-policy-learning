"""Vocabulary and target-window invariants.

These tests protect exposure accounting and causality at the data level: one
revealed target per window at its final valid step, deterministic prefix
clipping, the START token only at absolute episode start, and padding that
exists nowhere except collation.
"""

from __future__ import annotations

import pytest

# Reuse the codec test helpers for realistic arrays/sidecars (same directory,
# importable because pytest puts each test directory on sys.path).
from test_episode_codec import make_arrays, make_sidecar

from grounded_recovery.data import (
    ManifestError,
    Vocabulary,
    build_vocabulary,
    collate_windows,
    enumerate_windows,
    materialize_window,
    start_action_token,
)
from grounded_recovery.schemas import NULL_ACTION

VOCAB = build_vocabulary(["go to the red ball", "go to a blue key"])
NUM_ACTIONS = 3


def test_vocabulary_sorted_and_order_independent() -> None:
    shuffled = build_vocabulary(["go to a blue key", "go to the red ball"])
    assert shuffled.tokens == VOCAB.tokens
    assert shuffled.vocab_hash() == VOCAB.vocab_hash()
    assert VOCAB.tokens[0] == "<pad>"
    assert VOCAB.tokens[1] == "<unk>"
    assert list(VOCAB.tokens[2:]) == sorted(VOCAB.tokens[2:])


def test_vocabulary_unknown_word_maps_to_unk() -> None:
    unk = 1
    encoded = VOCAB.encode("go to the purple lava")
    assert encoded[3] == unk and encoded[4] == unk  # purple, lava unseen
    assert encoded[0] != unk  # go is in the vocabulary
    assert VOCAB.encode("go to the red ball") == VOCAB.encode("GO TO THE RED BALL")


def test_vocabulary_rejects_malformed() -> None:
    with pytest.raises(ManifestError):
        Vocabulary(tokens=("<unk>", "<pad>"))


def test_one_window_per_revealed_target() -> None:
    arrays = make_arrays(length=6, revealed=4)
    specs = enumerate_windows("ep", arrays, max_context_prefix=32, max_sequence_length=33)
    assert [spec.target_t for spec in specs] == [0, 1, 2, 3]
    assert all(spec.start_t == 0 for spec in specs)


def test_prefix_clip_deterministic_and_capped() -> None:
    arrays = make_arrays(length=10)
    specs = enumerate_windows("ep", arrays, max_context_prefix=3, max_sequence_length=4)
    # Early targets keep the full history; later ones are clipped to 4 steps.
    assert (specs[0].start_t, specs[0].target_t) == (0, 0)
    assert (specs[2].start_t, specs[2].target_t) == (0, 2)
    assert (specs[9].start_t, specs[9].target_t) == (6, 9)
    assert max(spec.length for spec in specs) == 4


def test_start_token_only_at_episode_start() -> None:
    arrays = make_arrays(length=10)
    sidecar = make_sidecar(arrays, "ep")
    specs = enumerate_windows("ep", arrays, max_context_prefix=3, max_sequence_length=4)
    start = start_action_token(NUM_ACTIONS)
    first = materialize_window(arrays, sidecar, specs[0], VOCAB, NUM_ACTIONS)
    assert first.prev_action_tokens[0] == start
    clipped = materialize_window(arrays, sidecar, specs[9], VOCAB, NUM_ACTIONS)
    # A clipped mid-episode window begins with the true executed action at
    # start_t - 1, never a fabricated START.
    assert clipped.prev_action_tokens[0] == int(arrays.executed_action[5])
    assert start not in clipped.prev_action_tokens


def test_corrupt_null_inside_episode_rejected() -> None:
    import dataclasses

    from grounded_recovery.schemas import EpisodeSchemaError

    arrays = make_arrays(length=6)
    bad_previous = arrays.previous_executed_action.copy()
    bad_previous[3] = NULL_ACTION
    # The schema layer itself rejects this array; the window layer would
    # reject it independently if it ever got through.
    with pytest.raises(EpisodeSchemaError, match="exactly at t=0"):
        dataclasses.replace(arrays, previous_executed_action=bad_previous)


def test_window_target_is_oracle_recommendation() -> None:
    arrays = make_arrays(length=5)
    sidecar = make_sidecar(arrays, "ep")
    spec = enumerate_windows("ep", arrays, 32, 33)[3]
    window = materialize_window(arrays, sidecar, spec, VOCAB, NUM_ACTIONS)
    assert window.target_action == int(arrays.oracle_recommended_action[3])


def test_window_reconstruction_checksum_stable() -> None:
    arrays = make_arrays(length=5)
    sidecar = make_sidecar(arrays, "ep")
    spec = enumerate_windows("ep", arrays, 32, 33)[2]
    first = materialize_window(arrays, sidecar, spec, VOCAB, NUM_ACTIONS)
    second = materialize_window(arrays, sidecar, spec, VOCAB, NUM_ACTIONS)
    assert first.checksum == second.checksum
    assert spec.source_step_ids() == ("ep:0", "ep:1", "ep:2")


def test_collation_masks_and_one_target() -> None:
    arrays = make_arrays(length=8)
    sidecar = make_sidecar(arrays, "ep")
    specs = enumerate_windows("ep", arrays, max_context_prefix=5, max_sequence_length=6)
    windows = [
        materialize_window(arrays, sidecar, specs[i], VOCAB, NUM_ACTIONS) for i in (0, 3, 7)
    ]
    batch = collate_windows(windows, channel_limits=(11, 6, 4))
    lengths = [w.spec.length for w in windows]
    assert batch.image.shape[1] == max(lengths)
    for row, length in enumerate(lengths):
        assert batch.step_mask[row].sum().item() == length
        assert batch.target_mask[row].sum().item() == 1
        assert bool(batch.target_mask[row, length - 1])
        # Padding regions are zeroed and masked out.
        assert not batch.step_mask[row, length:].any()
    assert batch.mission_lengths.tolist() == [
        len(VOCAB.encode(sidecar.mission))
    ] * 3


def test_collation_channel_range_guard() -> None:
    arrays = make_arrays(length=4)
    sidecar = make_sidecar(arrays, "ep")
    spec = enumerate_windows("ep", arrays, 32, 33)[0]
    window = materialize_window(arrays, sidecar, spec, VOCAB, NUM_ACTIONS)
    with pytest.raises(ManifestError, match="channel"):
        collate_windows([window], channel_limits=(2, 2, 2))
