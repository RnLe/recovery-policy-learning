"""Oracle synchronization contract (stub bot, no environment).

These tests protect label/state alignment: each oracle recommendation is only
a valid supervision target for the pre-action context it was computed in,
which requires exactly one ``replan`` per active step, fed with the executed
action. A double, skipped, or post-termination call would silently attach
labels to the wrong states.
"""

from __future__ import annotations

import pytest

from grounded_recovery import oracle as oracle_module
from grounded_recovery.config import EnvironmentConfig
from grounded_recovery.oracle import (
    OracleSupportError,
    OracleSyncError,
    SynchronizedOracle,
)

ENV_CFG = EnvironmentConfig(
    env_id="stub",
    doors_open=True,
    action_names=("left", "right", "forward"),
    action_ids=(0, 1, 2),
    bot_import="stub:StubBot",
    max_steps=10,
    oracle_recovery_gate=0.99,
)


class StubBot:
    """Scripted bot recording every ``replan`` argument."""

    def __init__(self, env: object, actions: tuple[int, ...] = (2, 2, 2, 2)) -> None:
        self.env = env
        self.actions = actions
        self.replan_args: list[int | None] = []

    def replan(self, action_taken: int | None = None) -> int:
        self.replan_args.append(action_taken)
        return self.actions[len(self.replan_args) - 1]


class StubSession:
    """Duck-typed WorldSession stand-in with controllable clock and doneness."""

    def __init__(self) -> None:
        self.cfg = ENV_CFG
        self.env = object()
        self.time = 0
        self.done = False


@pytest.fixture()
def stub_oracle(monkeypatch) -> tuple[SynchronizedOracle, StubSession, StubBot]:
    session = StubSession()
    created: list[StubBot] = []

    def fake_loader(bot_import: str) -> type:
        assert bot_import == ENV_CFG.bot_import

        def factory(env: object) -> StubBot:
            bot = StubBot(env)
            created.append(bot)
            return bot

        return factory

    monkeypatch.setattr(oracle_module, "load_bot_class", fake_loader)
    sync = SynchronizedOracle(session)  # type: ignore[arg-type]
    return sync, session, created[0]


def test_requires_freshly_reset_session(monkeypatch) -> None:
    session = StubSession()
    session.time = 3
    monkeypatch.setattr(oracle_module, "load_bot_class", lambda _: StubBot)
    with pytest.raises(OracleSyncError):
        SynchronizedOracle(session)  # type: ignore[arg-type]


def test_t0_sentinel_contract(stub_oracle) -> None:
    sync, _session, _bot = stub_oracle
    with pytest.raises(OracleSyncError):
        sync.recommend(0, 0)  # a concrete action at step 0 is a contract violation
    assert sync.recommend(None, 0) == 2
    with pytest.raises(OracleSyncError):
        sync.recommend(None, 1)  # the sentinel is only valid at step 0


def test_double_call_rejected(stub_oracle) -> None:
    sync, _session, _bot = stub_oracle
    sync.recommend(None, 0)
    with pytest.raises(OracleSyncError):
        sync.recommend(2, 0)


def test_skipped_index_rejected(stub_oracle) -> None:
    sync, _session, _bot = stub_oracle
    sync.recommend(None, 0)
    with pytest.raises(OracleSyncError):
        sync.recommend(2, 2)


def test_post_termination_call_rejected(stub_oracle) -> None:
    sync, session, _bot = stub_oracle
    sync.recommend(None, 0)
    session.done = True
    with pytest.raises(OracleSyncError):
        sync.recommend(2, 1)


def test_out_of_set_executed_action_rejected(stub_oracle) -> None:
    sync, _session, _bot = stub_oracle
    sync.recommend(None, 0)
    with pytest.raises(OracleSupportError):
        sync.recommend(5, 1)


def test_out_of_set_recommendation_rejected(monkeypatch) -> None:
    session = StubSession()

    def fake_loader(_: str) -> type:
        def factory(env: object) -> StubBot:
            return StubBot(env, actions=(6,))

        return factory

    monkeypatch.setattr(oracle_module, "load_bot_class", fake_loader)
    sync = SynchronizedOracle(session)  # type: ignore[arg-type]
    with pytest.raises(OracleSupportError, match="outside the frozen action set"):
        sync.recommend(None, 0)


def test_failed_call_does_not_advance_clock(stub_oracle) -> None:
    # A rejected query must leave the synchronization state unchanged, so the
    # caller cannot accidentally "recover" by repeating the bad call pattern.
    sync, _session, _bot = stub_oracle
    with pytest.raises(OracleSyncError):
        sync.recommend(2, 0)
    assert sync.next_step == 0
    assert sync.recommend(None, 0) == 2
    assert sync.next_step == 1


def test_replan_receives_executed_actions_verbatim(stub_oracle) -> None:
    sync, _session, bot = stub_oracle
    sync.recommend(None, 0)
    sync.recommend(1, 1)  # executed differs from the recommendation (2)
    sync.recommend(0, 2)
    assert bot.replan_args == [None, 1, 0]
    assert sync.calls == 3
