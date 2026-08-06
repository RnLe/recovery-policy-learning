"""Frozen experiment configuration: loading, validation, and contract hashing.

The YAML file is loaded into nested frozen dataclasses through explicit
per-section constructors. Unknown keys, missing keys, wrong types, and
out-of-range values are rejected with the dotted path of the offending field.
Fields that are deliberately undecided during the pilot carry the literal YAML
string ``PILOT_TO_FREEZE`` and load as ``None``; a configuration with status
``FROZEN`` must not contain any of them.

The contract hash is the SHA-256 of the canonical JSON form of the resolved
configuration and stamps every dataset, checkpoint, metric row, and manifest.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from grounded_recovery.artifacts import hash_json

PILOT_TO_FREEZE = "PILOT_TO_FREEZE"

STATUS_PILOT = "PILOT"
STATUS_FROZEN = "FROZEN"

SPLIT_NAMES: tuple[str, ...] = (
    "base",
    "collection",
    "validation",
    "operator_preflight",
    "test_candidate",
    "difficulty_shift",
    "expert_diagnostic",
    "visualization",
)


class ConfigError(ValueError):
    """The configuration file is malformed, incomplete, or inconsistent."""


@dataclass(frozen=True)
class StudyConfig:
    protocol_version: str
    status: str
    primary_contrast: str
    primary_endpoint: str
    sesoi_absolute_success: float


@dataclass(frozen=True)
class EnvironmentConfig:
    env_id: str
    doors_open: bool
    action_names: tuple[str, ...]
    action_ids: tuple[int, ...]
    bot_import: str
    max_steps: int
    oracle_recovery_gate: float


@dataclass(frozen=True)
class SplitCounts:
    base: int
    collection: int
    validation: int
    operator_preflight: int
    test_candidate: int
    difficulty_shift: int
    expert_diagnostic: int
    visualization: int

    def for_split(self, split: str) -> int:
        if split not in SPLIT_NAMES:
            raise ConfigError(f"unknown split name: {split!r}")
        return int(getattr(self, split))


@dataclass(frozen=True)
class DataConfig:
    n0: int | None
    b: int | None
    k: int | None
    h: int | None
    dataset_schema_version: str
    split_counts: SplitCounts


@dataclass(frozen=True)
class OperatorConfig:
    name: str
    mapping: tuple[int, ...]


@dataclass(frozen=True)
class PerturbationConfig:
    collection_operator: OperatorConfig
    unseen_operator: OperatorConfig
    collection_time_set: tuple[int, ...] | None
    unseen_time_set: tuple[int, ...] | None
    preflight_episodes_per_family: int
    preflight_time_min: int
    preflight_time_max: int


@dataclass(frozen=True)
class ModelConfig:
    num_objects: int
    num_colors: int
    num_states: int
    tile_embedding: int
    direction_embedding: int
    action_embedding: int
    conv_channels: int
    observation_projection: int
    word_embedding: int
    language_gru: int
    fusion: int
    policy_gru: int


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    base_updates: int
    base_targets_per_update: int
    new_targets_per_update: int | None
    updates_per_round: int | None
    max_context_prefix: int
    max_sequence_length: int
    sampling_with_replacement: bool
    optimizer_state_policy: str
    device: str


@dataclass(frozen=True)
class EvaluationConfig:
    desired_interval_half_width: float | None
    r_target: int | None
    r_max: int | None
    bootstrap_replicates: int


@dataclass(frozen=True)
class SeedConfig:
    root_seed: int
    bundle_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    study: StudyConfig
    environment: EnvironmentConfig
    data: DataConfig
    perturbation: PerturbationConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    seeds: SeedConfig


def _require_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
    return value


def _check_keys(section: Mapping[str, object], allowed: tuple[str, ...], path: str) -> None:
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ConfigError(f"{path}: unknown keys {unknown}")
    missing = sorted(set(allowed) - set(section))
    if missing:
        raise ConfigError(f"{path}: missing keys {missing}")


def _get(section: Mapping[str, object], key: str, path: str) -> object:
    if key not in section:
        raise ConfigError(f"{path}.{key}: missing")
    return section[key]


def _reject_placeholder(value: object, path: str) -> object:
    if value == PILOT_TO_FREEZE:
        raise ConfigError(f"{path}: {PILOT_TO_FREEZE} is not allowed for this field")
    return value


def _str(section: Mapping[str, object], key: str, path: str) -> str:
    value = _reject_placeholder(_get(section, key, path), f"{path}.{key}")
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{path}.{key}: expected a non-empty string")
    return value


def _bool(section: Mapping[str, object], key: str, path: str) -> bool:
    value = _reject_placeholder(_get(section, key, path), f"{path}.{key}")
    if not isinstance(value, bool):
        raise ConfigError(f"{path}.{key}: expected a boolean")
    return value


def _int(section: Mapping[str, object], key: str, path: str) -> int:
    value = _reject_placeholder(_get(section, key, path), f"{path}.{key}")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}.{key}: expected an integer")
    return value


def _float(section: Mapping[str, object], key: str, path: str) -> float:
    value = _reject_placeholder(_get(section, key, path), f"{path}.{key}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}.{key}: expected a number")
    return float(value)


def _int_tuple(section: Mapping[str, object], key: str, path: str) -> tuple[int, ...]:
    value = _reject_placeholder(_get(section, key, path), f"{path}.{key}")
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ConfigError(f"{path}.{key}: expected a list of integers")
    return tuple(value)


def _str_tuple(section: Mapping[str, object], key: str, path: str) -> tuple[str, ...]:
    value = _reject_placeholder(_get(section, key, path), f"{path}.{key}")
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigError(f"{path}.{key}: expected a list of strings")
    return tuple(value)


def _opt_float(section: Mapping[str, object], key: str, path: str) -> float | None:
    value = _get(section, key, path)
    if value == PILOT_TO_FREEZE:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}.{key}: expected a number or {PILOT_TO_FREEZE}")
    return float(value)


def _opt_int(section: Mapping[str, object], key: str, path: str) -> int | None:
    value = _get(section, key, path)
    if value == PILOT_TO_FREEZE:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}.{key}: expected an integer or {PILOT_TO_FREEZE}")
    return value


def _opt_int_tuple(section: Mapping[str, object], key: str, path: str) -> tuple[int, ...] | None:
    value = _get(section, key, path)
    if value == PILOT_TO_FREEZE:
        return None
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ConfigError(f"{path}.{key}: expected a list of integers or {PILOT_TO_FREEZE}")
    return tuple(value)


def _load_study(section: Mapping[str, object], path: str) -> StudyConfig:
    _check_keys(
        section,
        ("protocol_version", "status", "primary_contrast", "primary_endpoint",
         "sesoi_absolute_success"),
        path,
    )
    return StudyConfig(
        protocol_version=_str(section, "protocol_version", path),
        status=_str(section, "status", path),
        primary_contrast=_str(section, "primary_contrast", path),
        primary_endpoint=_str(section, "primary_endpoint", path),
        sesoi_absolute_success=_float(section, "sesoi_absolute_success", path),
    )


def _load_environment(section: Mapping[str, object], path: str) -> EnvironmentConfig:
    _check_keys(
        section,
        ("env_id", "doors_open", "action_names", "action_ids", "bot_import", "max_steps",
         "oracle_recovery_gate"),
        path,
    )
    return EnvironmentConfig(
        env_id=_str(section, "env_id", path),
        doors_open=_bool(section, "doors_open", path),
        action_names=_str_tuple(section, "action_names", path),
        action_ids=_int_tuple(section, "action_ids", path),
        bot_import=_str(section, "bot_import", path),
        max_steps=_int(section, "max_steps", path),
        oracle_recovery_gate=_float(section, "oracle_recovery_gate", path),
    )


def _load_split_counts(section: Mapping[str, object], path: str) -> SplitCounts:
    _check_keys(section, SPLIT_NAMES, path)
    return SplitCounts(**{name: _int(section, name, path) for name in SPLIT_NAMES})


def _load_data(section: Mapping[str, object], path: str) -> DataConfig:
    _check_keys(
        section,
        ("n0", "b", "k", "h", "dataset_schema_version", "split_counts"),
        path,
    )
    return DataConfig(
        n0=_opt_int(section, "n0", path),
        b=_opt_int(section, "b", path),
        k=_opt_int(section, "k", path),
        h=_opt_int(section, "h", path),
        dataset_schema_version=_str(section, "dataset_schema_version", path),
        split_counts=_load_split_counts(
            _require_mapping(_get(section, "split_counts", path), f"{path}.split_counts"),
            f"{path}.split_counts",
        ),
    )


def _load_operator(section: Mapping[str, object], path: str) -> OperatorConfig:
    _check_keys(section, ("name", "mapping"), path)
    return OperatorConfig(
        name=_str(section, "name", path),
        mapping=_int_tuple(section, "mapping", path),
    )


def _load_perturbation(section: Mapping[str, object], path: str) -> PerturbationConfig:
    _check_keys(
        section,
        ("collection_operator", "unseen_operator", "collection_time_set",
         "unseen_time_set", "preflight_episodes_per_family", "preflight_time_min",
         "preflight_time_max"),
        path,
    )
    return PerturbationConfig(
        collection_operator=_load_operator(
            _require_mapping(
                _get(section, "collection_operator", path), f"{path}.collection_operator"
            ),
            f"{path}.collection_operator",
        ),
        unseen_operator=_load_operator(
            _require_mapping(_get(section, "unseen_operator", path), f"{path}.unseen_operator"),
            f"{path}.unseen_operator",
        ),
        collection_time_set=_opt_int_tuple(section, "collection_time_set", path),
        unseen_time_set=_opt_int_tuple(section, "unseen_time_set", path),
        preflight_episodes_per_family=_int(section, "preflight_episodes_per_family", path),
        preflight_time_min=_int(section, "preflight_time_min", path),
        preflight_time_max=_int(section, "preflight_time_max", path),
    )


def _load_model(section: Mapping[str, object], path: str) -> ModelConfig:
    keys = (
        "num_objects", "num_colors", "num_states", "tile_embedding",
        "direction_embedding", "action_embedding", "conv_channels",
        "observation_projection", "word_embedding", "language_gru", "fusion",
        "policy_gru",
    )
    _check_keys(section, keys, path)
    return ModelConfig(**{key: _int(section, key, path) for key in keys})


def _load_training(section: Mapping[str, object], path: str) -> TrainingConfig:
    _check_keys(
        section,
        ("learning_rate", "weight_decay", "gradient_clip_norm", "base_updates",
         "base_targets_per_update", "new_targets_per_update", "updates_per_round",
         "max_context_prefix", "max_sequence_length", "sampling_with_replacement",
         "optimizer_state_policy", "device"),
        path,
    )
    return TrainingConfig(
        learning_rate=_float(section, "learning_rate", path),
        weight_decay=_float(section, "weight_decay", path),
        gradient_clip_norm=_float(section, "gradient_clip_norm", path),
        base_updates=_int(section, "base_updates", path),
        base_targets_per_update=_int(section, "base_targets_per_update", path),
        new_targets_per_update=_opt_int(section, "new_targets_per_update", path),
        updates_per_round=_opt_int(section, "updates_per_round", path),
        max_context_prefix=_int(section, "max_context_prefix", path),
        max_sequence_length=_int(section, "max_sequence_length", path),
        sampling_with_replacement=_bool(section, "sampling_with_replacement", path),
        optimizer_state_policy=_str(section, "optimizer_state_policy", path),
        device=_str(section, "device", path),
    )


def _load_evaluation(section: Mapping[str, object], path: str) -> EvaluationConfig:
    _check_keys(
        section,
        ("desired_interval_half_width", "r_target", "r_max", "bootstrap_replicates"),
        path,
    )
    return EvaluationConfig(
        desired_interval_half_width=_opt_float(section, "desired_interval_half_width", path),
        r_target=_opt_int(section, "r_target", path),
        r_max=_opt_int(section, "r_max", path),
        bootstrap_replicates=_int(section, "bootstrap_replicates", path),
    )


def _load_seeds(section: Mapping[str, object], path: str) -> SeedConfig:
    _check_keys(section, ("root_seed", "bundle_ids"), path)
    return SeedConfig(
        root_seed=_int(section, "root_seed", path),
        bundle_ids=_str_tuple(section, "bundle_ids", path),
    )


def load_config(path: Path) -> ExperimentConfig:
    """Load and structurally parse a configuration file (no semantic checks)."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    root = _require_mapping(raw, "config")
    _check_keys(
        root,
        ("study", "environment", "data", "perturbation", "model", "training",
         "evaluation", "seeds"),
        "config",
    )
    return ExperimentConfig(
        study=_load_study(_require_mapping(root["study"], "study"), "study"),
        environment=_load_environment(
            _require_mapping(root["environment"], "environment"), "environment"
        ),
        data=_load_data(_require_mapping(root["data"], "data"), "data"),
        perturbation=_load_perturbation(
            _require_mapping(root["perturbation"], "perturbation"), "perturbation"
        ),
        model=_load_model(_require_mapping(root["model"], "model"), "model"),
        training=_load_training(_require_mapping(root["training"], "training"), "training"),
        evaluation=_load_evaluation(
            _require_mapping(root["evaluation"], "evaluation"), "evaluation"
        ),
        seeds=_load_seeds(_require_mapping(root["seeds"], "seeds"), "seeds"),
    )


def unresolved_fields(cfg: ExperimentConfig) -> tuple[str, ...]:
    """Dotted paths of every field still carrying a pilot placeholder."""
    candidates = (
        ("data.n0", cfg.data.n0),
        ("data.b", cfg.data.b),
        ("data.k", cfg.data.k),
        ("data.h", cfg.data.h),
        ("perturbation.collection_time_set", cfg.perturbation.collection_time_set),
        ("perturbation.unseen_time_set", cfg.perturbation.unseen_time_set),
        ("training.new_targets_per_update", cfg.training.new_targets_per_update),
        ("training.updates_per_round", cfg.training.updates_per_round),
        ("evaluation.desired_interval_half_width",
         cfg.evaluation.desired_interval_half_width),
        ("evaluation.r_target", cfg.evaluation.r_target),
        ("evaluation.r_max", cfg.evaluation.r_max),
    )
    return tuple(name for name, value in candidates if value is None)


def _validate_operator(
    operator: OperatorConfig, action_ids: tuple[int, ...], path: str, errors: list[str]
) -> None:
    if len(operator.mapping) != len(action_ids):
        errors.append(
            f"{path}: mapping length {len(operator.mapping)} != action count {len(action_ids)}"
        )
        return
    if sorted(operator.mapping) != sorted(action_ids):
        errors.append(f"{path}: mapping is not a permutation of the action set {action_ids}")
        return
    for source, target in zip(action_ids, operator.mapping, strict=True):
        if source == target:
            errors.append(f"{path}: fixed point at action {source} (not a derangement)")


def validate_config(cfg: ExperimentConfig) -> None:
    """Reject semantically invalid configurations; raise ConfigError listing all violations."""
    errors: list[str] = []

    if cfg.study.status not in (STATUS_PILOT, STATUS_FROZEN):
        errors.append(
            f"study.status: must be {STATUS_PILOT} or {STATUS_FROZEN}, got {cfg.study.status!r}"
        )
    if cfg.study.status == STATUS_FROZEN:
        for name in unresolved_fields(cfg):
            errors.append(f"{name}: unresolved {PILOT_TO_FREEZE} value under status FROZEN")
    if not 0.0 < cfg.study.sesoi_absolute_success < 1.0:
        errors.append("study.sesoi_absolute_success: must be in (0, 1)")

    env = cfg.environment
    if len(env.action_names) != len(env.action_ids):
        errors.append("environment: action_names and action_ids must have equal length")
    if len(set(env.action_ids)) != len(env.action_ids):
        errors.append("environment.action_ids: duplicate ids")
    if len(set(env.action_names)) != len(env.action_names):
        errors.append("environment.action_names: duplicate names")
    if any(action_id < 0 or action_id > 6 for action_id in env.action_ids):
        errors.append("environment.action_ids: ids must lie in the MiniGrid action range 0..6")
    if ":" not in env.bot_import:
        errors.append("environment.bot_import: expected 'module.path:ClassName'")
    if env.max_steps < 1:
        errors.append("environment.max_steps: must be positive")
    if not 0.0 < env.oracle_recovery_gate <= 1.0:
        errors.append("environment.oracle_recovery_gate: must be in (0, 1]")

    data = cfg.data
    for split in SPLIT_NAMES:
        if data.split_counts.for_split(split) < 1:
            errors.append(f"data.split_counts.{split}: must be positive")
    if data.n0 is not None and data.n0 < 1:
        errors.append("data.n0: must be positive")
    if data.b is not None and data.b < 1:
        errors.append("data.b: must be positive")
    if data.k is not None and data.k < 1:
        errors.append("data.k: must be positive")
    if data.h is not None and data.h < 1:
        errors.append("data.h: must be positive")
    if data.b is not None and data.k is not None and data.b % data.k != 0:
        errors.append("data: b must be divisible by k (no remainder schedule is declared)")

    pert = cfg.perturbation
    _validate_operator(pert.collection_operator, env.action_ids, "perturbation.collection_operator",
                       errors)
    _validate_operator(pert.unseen_operator, env.action_ids, "perturbation.unseen_operator",
                       errors)
    if pert.collection_operator.mapping == pert.unseen_operator.mapping:
        errors.append(
            "perturbation: collection and unseen operators must be distinct derangements"
        )
    if pert.preflight_episodes_per_family < 1:
        errors.append("perturbation.preflight_episodes_per_family: must be positive")
    if not 0 <= pert.preflight_time_min <= pert.preflight_time_max:
        errors.append("perturbation: require 0 <= preflight_time_min <= preflight_time_max")
    if pert.preflight_time_max >= env.max_steps:
        errors.append("perturbation.preflight_time_max: must be below environment.max_steps")

    model = cfg.model
    for field in dataclasses.fields(model):
        if getattr(model, field.name) < 1:
            errors.append(f"model.{field.name}: must be positive")

    training = cfg.training
    if training.learning_rate <= 0.0:
        errors.append("training.learning_rate: must be positive")
    if training.weight_decay < 0.0:
        errors.append("training.weight_decay: must be non-negative")
    if training.gradient_clip_norm <= 0.0:
        errors.append("training.gradient_clip_norm: must be positive")
    if training.base_updates < 1:
        errors.append("training.base_updates: must be positive")
    if training.base_targets_per_update < 1:
        errors.append("training.base_targets_per_update: must be positive")
    if training.new_targets_per_update is not None and training.new_targets_per_update < 1:
        errors.append("training.new_targets_per_update: must be positive")
    if training.updates_per_round is not None and training.updates_per_round < 1:
        errors.append("training.updates_per_round: must be positive")
    if training.max_context_prefix < 0:
        errors.append("training.max_context_prefix: must be non-negative")
    if training.max_sequence_length < training.max_context_prefix + 1:
        errors.append(
            "training.max_sequence_length: must be at least max_context_prefix + 1 "
            "(the window is the prefix plus the target step)"
        )
    if training.optimizer_state_policy not in ("continue", "reset"):
        errors.append("training.optimizer_state_policy: must be 'continue' or 'reset'")
    if training.device not in ("cpu", "cuda"):
        errors.append("training.device: must be 'cpu' or 'cuda'")

    evaluation = cfg.evaluation
    if (
        evaluation.desired_interval_half_width is not None
        and not 0.0 < evaluation.desired_interval_half_width < 1.0
    ):
        errors.append("evaluation.desired_interval_half_width: must be in (0, 1)")
    if evaluation.r_target is not None and evaluation.r_target < 5:
        errors.append(
            "evaluation.r_target: at least five complete pipeline bundles are the "
            "hard minimum for confirmatory labeling"
        )
    if evaluation.r_max is not None and evaluation.r_max < 1:
        errors.append("evaluation.r_max: must be positive")
    if evaluation.bootstrap_replicates < 100:
        errors.append("evaluation.bootstrap_replicates: must be at least 100")

    seeds = cfg.seeds
    if seeds.root_seed < 0:
        errors.append("seeds.root_seed: must be non-negative")
    if not seeds.bundle_ids:
        errors.append("seeds.bundle_ids: must not be empty")
    if len(set(seeds.bundle_ids)) != len(seeds.bundle_ids):
        errors.append("seeds.bundle_ids: duplicate bundle ids")

    if errors:
        raise ConfigError("invalid configuration:\n  " + "\n  ".join(errors))


def load_and_validate(path: Path) -> ExperimentConfig:
    cfg = load_config(path)
    validate_config(cfg)
    return cfg


def config_as_dict(cfg: ExperimentConfig) -> dict[str, object]:
    return dataclasses.asdict(cfg)


def contract_hash(cfg: ExperimentConfig) -> str:
    """SHA-256 identity of the resolved configuration."""
    return hash_json(config_as_dict(cfg))


def scenario_identity_hash(cfg: ExperimentConfig) -> str:
    """SHA-256 identity of everything that determines generated scenarios.

    Manifests are a function of the environment, the root seed, the split
    counts, the dataset schema, and the preflight scheduling window, not of
    training hyperparameters. Keying manifest validity to this subset lets the
    pilot tune learning settings without regenerating (or worse, silently
    invalidating) the frozen scenario panels. The full contract hash is still
    recorded on every manifest for provenance.
    """
    return hash_json(
        {
            "environment": dataclasses.asdict(cfg.environment),
            "root_seed": cfg.seeds.root_seed,
            "split_counts": dataclasses.asdict(cfg.data.split_counts),
            "dataset_schema_version": cfg.data.dataset_schema_version,
            "preflight_time_min": cfg.perturbation.preflight_time_min,
            "preflight_time_max": cfg.perturbation.preflight_time_max,
        }
    )
