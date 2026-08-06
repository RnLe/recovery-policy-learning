"""Environment adapter for the frozen BabyAI task.

``WorldSession`` owns exactly one Gymnasium environment and is the only code
that calls ``reset``/``step``. Every reset requires an explicit seed (a reset
without a seed would silently continue the previous RNG stream), every step
validates the frozen action set, and stepping after termination is an error.
The reset world can be captured as a ``ScenarioState`` for canonical scenario
hashing.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import struct
from collections.abc import Sequence
from dataclasses import dataclass

import gymnasium as gym
import minigrid  # noqa: F401  # importing minigrid registers the BabyAI environments
import numpy as np

from grounded_recovery.config import EnvironmentConfig

OBSERVATION_IMAGE_SHAPE = (7, 7, 3)
DIRECTION_RANGE = range(4)


class WorldError(RuntimeError):
    """The environment was driven outside its frozen contract."""


@dataclass(frozen=True)
class StepResult:
    """One observation plus the transition outcome that produced it."""

    image: np.ndarray
    direction: int
    mission: str
    reward: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class ScenarioState:
    """Canonical serialization of a reset world, independent of the seed."""

    env_id: str
    grid_encoding: tuple[tuple[tuple[int, ...], ...], ...]
    agent_pos: tuple[int, int]
    agent_dir: int
    mission: str


def assert_observation_schema(obs: object) -> None:
    if not isinstance(obs, dict):
        raise WorldError(f"observation must be a dict, got {type(obs).__name__}")
    for key in ("image", "direction", "mission"):
        if key not in obs:
            raise WorldError(f"observation missing key {key!r}")
    image = obs["image"]
    if not isinstance(image, np.ndarray) or image.shape != OBSERVATION_IMAGE_SHAPE:
        raise WorldError(f"observation image must be ndarray {OBSERVATION_IMAGE_SHAPE}")
    if image.dtype != np.uint8:
        raise WorldError(f"observation image dtype must be uint8, got {image.dtype}")
    direction = obs["direction"]
    if isinstance(direction, bool) or not isinstance(direction, (int, np.integer)):
        raise WorldError("observation direction must be an integer")
    if int(direction) not in DIRECTION_RANGE:
        raise WorldError(f"observation direction out of range: {direction}")
    if not isinstance(obs["mission"], str) or not obs["mission"]:
        raise WorldError("observation mission must be a non-empty string")


class WorldSession:
    """Holder for one environment, its clock, and its termination state."""

    def __init__(self, cfg: EnvironmentConfig, render_mode: str | None = None) -> None:
        self.cfg = cfg
        self.env = gym.make(
            cfg.env_id,
            disable_env_checker=True,
            doors_open=cfg.doors_open,
            render_mode=render_mode,
        )
        self._action_ids = frozenset(cfg.action_ids)
        self._time: int | None = None
        self._done = False
        self._last: StepResult | None = None

    @property
    def time(self) -> int:
        if self._time is None:
            raise WorldError("session has not been reset")
        return self._time

    @property
    def done(self) -> bool:
        return self._done

    @property
    def last_observation(self) -> StepResult:
        if self._last is None:
            raise WorldError("session has not been reset")
        return self._last

    def reset(self, seed: int) -> StepResult:
        obs, _info = self.env.reset(seed=int(seed))
        assert_observation_schema(obs)
        # max_steps is derived by the environment during reset; the frozen
        # contract value must hold for every generated world.
        actual_max_steps = int(self.env.unwrapped.max_steps)
        if actual_max_steps != self.cfg.max_steps:
            raise WorldError(
                f"environment max_steps {actual_max_steps} does not match the "
                f"frozen contract value {self.cfg.max_steps}"
            )
        result = StepResult(
            image=np.array(obs["image"], dtype=np.uint8, copy=True),
            direction=int(obs["direction"]),
            mission=str(obs["mission"]),
            reward=0.0,
            terminated=False,
            truncated=False,
        )
        self._time = 0
        self._done = False
        self._last = result
        return result

    def step(self, action: int) -> StepResult:
        if self._time is None:
            raise WorldError("step before reset")
        if self._done:
            raise WorldError("step after termination")
        if isinstance(action, bool) or int(action) not in self._action_ids:
            raise WorldError(
                f"action {action!r} is outside the frozen action set {sorted(self._action_ids)}"
            )
        obs, reward, terminated, truncated, _info = self.env.step(int(action))
        assert_observation_schema(obs)
        result = StepResult(
            image=np.array(obs["image"], dtype=np.uint8, copy=True),
            direction=int(obs["direction"]),
            mission=str(obs["mission"]),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        self._time += 1
        self._done = result.terminated or result.truncated
        self._last = result
        return result

    def scenario_state(self) -> ScenarioState:
        """Capture the reset world; only valid before the first step."""
        if self._time != 0:
            raise WorldError("scenario_state is only defined at t=0, immediately after reset")
        unwrapped = self.env.unwrapped
        grid = np.asarray(unwrapped.grid.encode())
        grid_tuple = tuple(
            tuple(tuple(int(channel) for channel in cell) for cell in column) for column in grid
        )
        position = unwrapped.agent_pos
        return ScenarioState(
            env_id=self.cfg.env_id,
            grid_encoding=grid_tuple,
            agent_pos=(int(position[0]), int(position[1])),
            agent_dir=int(unwrapped.agent_dir),
            mission=self.last_observation.mission,
        )

    def fingerprint(self) -> dict[str, object]:
        """Machine-checkable identity of the runtime environment."""
        packages = {}
        for name in ("minigrid", "gymnasium", "numpy", "torch"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        return {
            "env_id": self.cfg.env_id,
            "doors_open": self.cfg.doors_open,
            "action_space": repr(self.env.action_space),
            "observation_space_keys": sorted(self.env.observation_space.spaces.keys()),
            "max_steps": int(self.env.unwrapped.max_steps),
            "frozen_action_ids": list(self.cfg.action_ids),
            "frozen_action_names": list(self.cfg.action_names),
            "packages": packages,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "babyai_done_actions_env_var": os.environ.get("BABYAI_DONE_ACTIONS"),
        }

    def render_frame(self) -> np.ndarray:
        """Full-grid RGB frame (requires render_mode="rgb_array")."""
        frame = self.env.render()
        if frame is None:
            raise WorldError('render_frame requires render_mode="rgb_array"')
        return np.asarray(frame)

    def close(self) -> None:
        self.env.close()


def replay_episode(
    cfg: EnvironmentConfig, seed: int, actions: Sequence[int]
) -> list[StepResult]:
    """Reset from ``seed`` and execute ``actions``; error if they overrun termination."""
    session = WorldSession(cfg)
    try:
        results = [session.reset(seed)]
        for index, action in enumerate(actions):
            if session.done:
                raise WorldError(
                    f"replay actions extend past termination at index {index}"
                )
            results.append(session.step(action))
        return results
    finally:
        session.close()


def trace_digest(results: Sequence[StepResult]) -> str:
    """Order-sensitive digest of a trace; byte-exact over images and outcomes."""
    digest = hashlib.sha256()
    for result in results:
        digest.update(result.image.tobytes())
        digest.update(struct.pack(">B", result.direction))
        digest.update(struct.pack(">d", result.reward))
        digest.update(struct.pack(">??", result.terminated, result.truncated))
        digest.update(result.mission.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()
