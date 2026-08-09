"""Purpose-split manifests: probing, disjointness auditing, and storage.

Each of the eight purpose splits is a deterministic function of the contract:
candidate seeds come from the split's named seed stream, each candidate is
probed by a full nominal oracle rollout, and inadmissible or colliding
candidates are skipped by a deterministic skip-and-advance rule with every
rejection recorded. Split disjointness is asserted over both environment
seeds and canonical scenario hashes.

Only the ``operator_preflight`` split carries concrete scheduled intervention
times at ``manifest_version`` 1; the frozen time sets of the other splits are
added later as a new manifest version with identical seeds and hashes.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from grounded_recovery.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    hash_json,
    read_json,
    read_jsonl,
)
from grounded_recovery.config import (
    SPLIT_NAMES,
    ExperimentConfig,
    contract_hash,
    scenario_identity_hash,
)
from grounded_recovery.oracle import OracleSupportError, run_synchronized_episode
from grounded_recovery.schemas import (
    EPISODE_ARRAY_FIELDS,
    MANIFEST_VERSION,
    NULL_ACTION,
    SCENARIO_HASH_VERSION,
    EpisodeArrays,
    EpisodeSidecar,
    ManifestEntry,
    episode_content_checksum,
    manifest_entry_from_json,
    manifest_entry_to_json,
    sidecar_from_json,
    sidecar_to_json,
)
from grounded_recovery.seeds import derive_seed, seed_stream
from grounded_recovery.world import WorldSession

if TYPE_CHECKING:
    import torch

# Seed streams for manifests are global to the experiment, not per pipeline
# bundle: every bundle trains and evaluates on the same frozen scenario splits.
GLOBAL_BUNDLE = "global"

# Candidate seeds consumed per accepted entry before generation is declared
# stuck (guards against an infinite skip-and-advance loop).
MAX_CANDIDATE_FACTOR = 50


class ManifestError(RuntimeError):
    """Manifest generation, storage, or verification failed."""


@dataclass(frozen=True)
class ScenarioProbe:
    """Outcome of one nominal oracle rollout on one candidate seed."""

    seed: int
    scenario_hash: str
    mission: str
    path_length: int
    admissible: bool
    reason: str | None


def probe_scenario(session: WorldSession, seed: int) -> ScenarioProbe:
    """Probe one candidate seed with a full nominal oracle rollout."""
    try:
        trace = run_synchronized_episode(session, seed, lambda t, rec: rec)
    except OracleSupportError as error:
        # Trivial or degenerate start: the bot answered outside the frozen
        # movement set (typically `done` on an already-satisfied mission).
        return ScenarioProbe(
            seed=seed,
            scenario_hash="",
            mission="",
            path_length=0,
            admissible=False,
            reason=f"oracle_support: {error}",
        )
    if not trace.success:
        return ScenarioProbe(
            seed=seed,
            scenario_hash=trace.scenario_hash,
            mission=trace.mission,
            path_length=len(trace.transitions),
            admissible=False,
            reason="oracle_failure",
        )
    return ScenarioProbe(
        seed=seed,
        scenario_hash=trace.scenario_hash,
        mission=trace.mission,
        path_length=len(trace.transitions),
        admissible=True,
        reason=None,
    )


def _preflight_schedule_time(cfg: ExperimentConfig, ordinal: int, path_length: int) -> int:
    """Deterministic scheduled corruption time for one preflight scenario.

    Drawn uniformly from ``[preflight_time_min, min(preflight_time_max,
    path_length - 1)]``. The upper bound conditions on the scenario's fixed
    nominal path length so every preflight corruption is deliverable; this is
    measurement design for the oracle-recovery gate, recorded in the report.
    """
    low = cfg.perturbation.preflight_time_min
    high = min(cfg.perturbation.preflight_time_max, path_length - 1)
    if high < low:
        raise ManifestError(
            f"preflight scenario with path length {path_length} cannot schedule a "
            f"corruption at or after t={low}"
        )
    raw = derive_seed(
        cfg.seeds.root_seed, GLOBAL_BUNDLE, f"manifest.operator_preflight.schedule.{ordinal}"
    )
    return low + raw % (high - low + 1)


def _split_admissible(cfg: ExperimentConfig, split: str, probe: ScenarioProbe) -> str | None:
    """Extra per-split admissibility beyond nominal oracle success."""
    if probe.path_length < 1:
        return "trivial"
    if split == "operator_preflight":
        if probe.path_length < cfg.perturbation.preflight_time_min + 1:
            return "too_short_for_preflight_schedule"
    return None


def build_split_manifest(
    cfg: ExperimentConfig,
    session: WorldSession,
    split: str,
    claimed_seeds: set[int],
    claimed_hashes: set[str],
) -> tuple[list[ManifestEntry], list[ScenarioProbe]]:
    """Build one split manifest, consuming its named seed stream deterministically."""
    if split not in SPLIT_NAMES:
        raise ManifestError(f"unknown split {split!r}")
    count = cfg.data.split_counts.for_split(split)
    entries: list[ManifestEntry] = []
    rejects: list[ScenarioProbe] = []
    stream = seed_stream(cfg.seeds.root_seed, GLOBAL_BUNDLE, f"manifest.{split}")
    candidates = 0
    while len(entries) < count:
        candidates += 1
        if candidates > count * MAX_CANDIDATE_FACTOR:
            raise ManifestError(
                f"split {split}: exceeded {count * MAX_CANDIDATE_FACTOR} candidate seeds "
                f"for {count} entries; the environment or admissibility rule is degenerate"
            )
        seed = next(stream)
        if seed in claimed_seeds:
            rejects.append(
                ScenarioProbe(seed, "", "", 0, False, "seed_collision")
            )
            continue
        probe = probe_scenario(session, seed)
        if probe.admissible and probe.scenario_hash in claimed_hashes:
            probe = ScenarioProbe(
                probe.seed, probe.scenario_hash, probe.mission, probe.path_length,
                False, "scenario_hash_collision",
            )
        reason = probe.reason
        if probe.admissible:
            reason = _split_admissible(cfg, split, probe)
        if reason is not None:
            rejects.append(
                ScenarioProbe(
                    probe.seed, probe.scenario_hash, probe.mission, probe.path_length,
                    False, reason,
                )
            )
            continue
        ordinal = len(entries)
        scheduled: tuple[int, ...] = ()
        family: str | None = None
        if split == "operator_preflight":
            scheduled = (_preflight_schedule_time(cfg, ordinal, probe.path_length),)
        entries.append(
            ManifestEntry(
                split_name=split,
                ordinal=ordinal,
                environment_id=cfg.environment.env_id,
                environment_seed=probe.seed,
                canonical_scenario_hash=probe.scenario_hash,
                mission=probe.mission,
                nominal_oracle_path_length=probe.path_length,
                perturbation_family=family,
                scheduled_intervention_times=scheduled,
                manifest_version=MANIFEST_VERSION,
            )
        )
        claimed_seeds.add(probe.seed)
        claimed_hashes.add(probe.scenario_hash)
    return entries, rejects


def audit_disjointness(
    manifests: dict[str, list[ManifestEntry]],
) -> dict[str, object]:
    """Assert pairwise seed and scenario-hash disjointness across all splits."""
    seed_sets = {split: {e.environment_seed for e in entries} for split, entries in
                 manifests.items()}
    hash_sets = {split: {e.canonical_scenario_hash for e in entries} for split, entries in
                 manifests.items()}
    overlaps: list[dict[str, object]] = []
    names = sorted(manifests)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            seed_overlap = seed_sets[first] & seed_sets[second]
            hash_overlap = hash_sets[first] & hash_sets[second]
            if seed_overlap or hash_overlap:
                overlaps.append(
                    {
                        "splits": [first, second],
                        "seed_overlap": sorted(seed_overlap),
                        "hash_overlap": sorted(hash_overlap),
                    }
                )
    report = {
        "splits": {
            split: {"entries": len(manifests[split]), "unique_seeds": len(seed_sets[split]),
                    "unique_hashes": len(hash_sets[split])}
            for split in names
        },
        "pairwise_overlaps": overlaps,
        "disjoint": not overlaps,
    }
    if overlaps:
        raise ManifestError(f"split disjointness violated: {overlaps}")
    return report


def write_split_manifest(
    manifest_root: Path,
    cfg: ExperimentConfig,
    split: str,
    entries: list[ManifestEntry],
    rejects: list[ScenarioProbe],
) -> str:
    split_dir = Path(manifest_root) / split
    entries_path = split_dir / "entries.jsonl"
    manifest_hash = atomic_write_jsonl(
        entries_path, (manifest_entry_to_json(entry) for entry in entries)
    )
    reject_counts: dict[str, int] = {}
    for reject in rejects:
        key = (reject.reason or "unknown").split(":")[0]
        reject_counts[key] = reject_counts.get(key, 0) + 1
    atomic_write_json(
        split_dir / "manifest_meta.json",
        {
            "split_name": split,
            "entry_count": len(entries),
            "manifest_version": MANIFEST_VERSION,
            "scenario_hash_version": SCENARIO_HASH_VERSION,
            "manifest_hash": manifest_hash,
            "scenario_identity_hash": scenario_identity_hash(cfg),
            "contract_hash": contract_hash(cfg),
            "rejected_candidates": len(rejects),
            "rejected_by_reason": reject_counts,
        },
    )
    atomic_write_jsonl(split_dir / "rejected_probes.jsonl", (asdict(r) for r in rejects))
    return manifest_hash


def load_split_manifest(manifest_root: Path, split: str) -> tuple[list[ManifestEntry], str]:
    """Load one split manifest and verify its stored hash against the file bytes."""
    split_dir = Path(manifest_root) / split
    meta = read_json(split_dir / "manifest_meta.json")
    entries_path = split_dir / "entries.jsonl"
    actual_hash = file_sha256(entries_path)
    if actual_hash != meta["manifest_hash"]:
        raise ManifestError(
            f"manifest {split}: entries file hash {actual_hash} does not match "
            f"recorded hash {meta['manifest_hash']}"
        )
    entries = [manifest_entry_from_json(row) for row in read_jsonl(entries_path)]
    if len(entries) != meta["entry_count"]:
        raise ManifestError(
            f"manifest {split}: {len(entries)} entries, meta records {meta['entry_count']}"
        )
    for expected_ordinal, entry in enumerate(entries):
        if entry.ordinal != expected_ordinal:
            raise ManifestError(f"manifest {split}: ordinal gap at {expected_ordinal}")
    return entries, actual_hash


def verify_manifest_contract(manifest_root: Path, split: str, cfg: ExperimentConfig) -> None:
    """Manifest validity is keyed to the scenario-identity subset of the contract.

    Training hyperparameters may change during the pilot without touching the
    generated worlds; anything that could change the scenarios themselves
    (environment, root seed, split counts, schema, preflight window) must
    match exactly.
    """
    meta = read_json(Path(manifest_root) / split / "manifest_meta.json")
    expected = scenario_identity_hash(cfg)
    if meta["scenario_identity_hash"] != expected:
        raise ManifestError(
            f"manifest {split} was generated under scenario identity "
            f"{meta['scenario_identity_hash'][:12]}..., current identity is "
            f"{expected[:12]}...; regenerate manifests"
        )


def make_manifests(cfg: ExperimentConfig, manifest_root: Path) -> dict[str, str]:
    """Generate all eight split manifests, audit disjointness, and write them."""
    manifest_root = Path(manifest_root)
    for split in SPLIT_NAMES:
        if (manifest_root / split / "entries.jsonl").exists():
            raise ManifestError(
                f"manifest for split {split!r} already exists under {manifest_root}; "
                "manifests are immutable; remove or version them deliberately"
            )
    session = WorldSession(cfg.environment)
    try:
        claimed_seeds: set[int] = set()
        claimed_hashes: set[str] = set()
        manifests: dict[str, list[ManifestEntry]] = {}
        all_rejects: dict[str, list[ScenarioProbe]] = {}
        for split in SPLIT_NAMES:
            entries, rejects = build_split_manifest(
                cfg, session, split, claimed_seeds, claimed_hashes
            )
            manifests[split] = entries
            all_rejects[split] = rejects
    finally:
        session.close()
    report = audit_disjointness(manifests)
    hashes: dict[str, str] = {}
    for split in SPLIT_NAMES:
        hashes[split] = write_split_manifest(
            manifest_root, cfg, split, manifests[split], all_rejects[split]
        )
    atomic_write_json(
        manifest_root / "disjointness_report.json",
        {
            "contract_hash": contract_hash(cfg),
            "manifest_hashes": hashes,
            "audit": report,
        },
    )
    return hashes


# --- Episode storage ---------------------------------------------------------

def episode_paths(episodes_dir: Path, episode_id: str) -> tuple[Path, Path]:
    episodes_dir = Path(episodes_dir)
    return episodes_dir / f"{episode_id}.npz", episodes_dir / f"{episode_id}.json"


def write_episode(episodes_dir: Path, arrays: EpisodeArrays, sidecar: EpisodeSidecar) -> None:
    """Store one episode as compressed arrays plus a JSON sidecar, atomically."""
    npz_path, sidecar_path = episode_paths(episodes_dir, sidecar.episode_id)
    if npz_path.exists() or sidecar_path.exists():
        raise ManifestError(f"episode {sidecar.episode_id} already exists in {episodes_dir}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz_path.with_name(npz_path.name + ".tmp.npz")
    try:
        with open(tmp, "wb") as handle:
            np.savez_compressed(
                handle,
                **{name: getattr(arrays, name) for name in EPISODE_ARRAY_FIELDS},
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, npz_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    atomic_write_json(sidecar_path, sidecar_to_json(sidecar))


def read_episode(
    episodes_dir: Path, episode_id: str
) -> tuple[EpisodeArrays, EpisodeSidecar]:
    """Load one stored episode and verify its content checksum."""
    npz_path, sidecar_path = episode_paths(episodes_dir, episode_id)
    sidecar = sidecar_from_json(read_json(sidecar_path))
    with np.load(npz_path, allow_pickle=False) as archive:
        stored_fields = set(archive.files)
        expected_fields = set(EPISODE_ARRAY_FIELDS)
        if stored_fields != expected_fields:
            raise ManifestError(
                f"episode {episode_id}: stored arrays {sorted(stored_fields)} do not match "
                f"the schema fields"
            )
        arrays = EpisodeArrays(**{name: archive[name] for name in EPISODE_ARRAY_FIELDS})
    identity = {
        "episode_id": sidecar.episode_id,
        "environment_seed": sidecar.environment_seed,
        "canonical_scenario_hash": sidecar.canonical_scenario_hash,
        "mission": sidecar.mission,
        "source_arm": sidecar.source_arm,
        "round_index": sidecar.round_index,
    }
    actual = episode_content_checksum(arrays, identity)
    if actual != sidecar.content_checksum:
        raise ManifestError(
            f"episode {episode_id}: content checksum mismatch "
            f"(stored {sidecar.content_checksum[:12]}..., recomputed {actual[:12]}...)"
        )
    return arrays, sidecar


# --- Base demonstration collection ------------------------------------------

@dataclass(frozen=True)
class BaseCollectionSummary:
    bundle_id: str
    n0: int
    episodes: int
    steps: int
    oracle_calls: int
    final_episode_budget_truncated: bool
    dataset_dir: str
    ledger_final_hash: str


def base_dataset_dir(cfg: ExperimentConfig, bundle_id: str, data_root: Path) -> Path:
    return Path(data_root) / contract_hash(cfg)[:12] / bundle_id / "base"


def collect_base(
    cfg: ExperimentConfig,
    bundle_id: str,
    manifest_root: Path,
    data_root: Path,
) -> BaseCollectionSummary:
    """Collect nominal oracle trajectories until exactly ``N0`` revealed targets.

    Episodes are collected in base-manifest order; every active step's oracle
    recommendation is a revealed target. When the budget boundary falls inside
    an episode, stepping stops immediately after the transition that consumed
    the N0-th target, and the episode is stored budget-truncated with its
    preceding context intact. No later manifest entry is consumed.
    """
    from grounded_recovery.integrity import LedgerWriter, recount_dataset
    from grounded_recovery.schemas import episode_from_trace

    if cfg.data.n0 is None:
        raise ManifestError("data.n0 is unresolved (PILOT_TO_FREEZE); cannot collect base data")
    if bundle_id not in cfg.seeds.bundle_ids:
        raise ManifestError(f"bundle {bundle_id!r} is not declared in seeds.bundle_ids")
    n0 = int(cfg.data.n0)

    verify_manifest_contract(manifest_root, "base", cfg)
    entries, manifest_hash = load_split_manifest(manifest_root, "base")

    dataset_dir = base_dataset_dir(cfg, bundle_id, data_root)
    if dataset_dir.exists():
        raise ManifestError(
            f"base dataset already exists at {dataset_dir}; datasets are immutable"
        )
    episodes_dir = dataset_dir / "episodes"
    cfg_hash = contract_hash(cfg)

    session = WorldSession(cfg.environment)
    ledger = LedgerWriter(dataset_dir / "collection_ledger.jsonl")
    index_rows: list[dict[str, object]] = []
    remaining = n0
    total_steps = 0
    total_oracle_calls = 0
    final_truncated_by_budget = False
    try:
        for entry in entries:
            if remaining == 0:
                break
            budget = remaining

            def stop(completed: int, _budget: int = budget) -> bool:
                return completed >= _budget

            trace = run_synchronized_episode(
                session, entry.environment_seed, lambda t, rec: rec, stop_after_step=stop
            )
            if trace.scenario_hash != entry.canonical_scenario_hash:
                raise ManifestError(
                    f"base collection replay of seed {entry.environment_seed} produced "
                    f"scenario hash {trace.scenario_hash}, manifest records "
                    f"{entry.canonical_scenario_hash}"
                )
            length = len(trace.transitions)
            if length > remaining:
                raise ManifestError(
                    f"collector overshot the target budget on seed {entry.environment_seed}"
                )
            if trace.stopped_early:
                termination_reason = "budget_truncated"
                final_truncated_by_budget = True
            elif trace.success:
                termination_reason = "terminated"
            else:
                termination_reason = "truncated"
            episode_id = f"base_{entry.ordinal:05d}"
            arrays, sidecar = episode_from_trace(
                trace,
                episode_id=episode_id,
                reveal_mask=np.ones(length, dtype=np.bool_),
                source_arm="base",
                round_index=0,
                termination_reason=termination_reason,
                intervention=None,
                contract_hash=cfg_hash,
                manifest_hash=manifest_hash,
            )
            write_episode(episodes_dir, arrays, sidecar)
            ledger.append(
                episode_id=episode_id,
                episode_targets=arrays.revealed_targets,
                episode_steps=length,
                oracle_calls=trace.oracle_calls,
                budget_truncated=trace.stopped_early,
                episode_checksum=sidecar.content_checksum,
            )
            index_rows.append(
                {
                    "episode_id": episode_id,
                    "environment_seed": entry.environment_seed,
                    "canonical_scenario_hash": entry.canonical_scenario_hash,
                    "revealed_targets": arrays.revealed_targets,
                    "content_checksum": sidecar.content_checksum,
                }
            )
            remaining -= arrays.revealed_targets
            total_steps += length
            total_oracle_calls += trace.oracle_calls
        if remaining > 0:
            raise ManifestError(
                f"base manifest exhausted with {remaining} of {n0} targets uncollected; "
                "increase the base split count during pilot design"
            )
    finally:
        session.close()
        ledger.close()

    atomic_write_jsonl(dataset_dir / "episode_index.jsonl", iter(index_rows))
    final_hash = ledger.finalize(contract_hash=cfg_hash, manifest_hash=manifest_hash)
    atomic_write_json(
        dataset_dir / "dataset_meta.json",
        {
            "bundle_id": bundle_id,
            "source_arm": "base",
            "n0": n0,
            "episodes": len(index_rows),
            "steps": total_steps,
            "oracle_calls": total_oracle_calls,
            "contract_hash": cfg_hash,
            "manifest_hash": manifest_hash,
            "ledger_final_hash": final_hash,
            "dataset_schema_version": cfg.data.dataset_schema_version,
        },
    )
    recount = recount_dataset(dataset_dir)
    if recount["targets"] != n0:
        raise ManifestError(
            f"recount found {recount['targets']} targets, exactly {n0} were required"
        )
    return BaseCollectionSummary(
        bundle_id=bundle_id,
        n0=n0,
        episodes=len(index_rows),
        steps=total_steps,
        oracle_calls=total_oracle_calls,
        final_episode_budget_truncated=final_truncated_by_budget,
        dataset_dir=str(dataset_dir),
        ledger_final_hash=final_hash,
    )


# --- Mission vocabulary ------------------------------------------------------

PAD_TOKEN = 0
UNK_TOKEN = 1


@dataclass(frozen=True)
class Vocabulary:
    """Mission token vocabulary; index equals token id."""

    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.tokens) < 2 or self.tokens[0] != "<pad>" or self.tokens[1] != "<unk>":
            raise ManifestError("vocabulary must start with <pad>, <unk>")
        if len(set(self.tokens)) != len(self.tokens):
            raise ManifestError("vocabulary contains duplicate tokens")

    @property
    def size(self) -> int:
        return len(self.tokens)

    def encode(self, mission: str) -> tuple[int, ...]:
        index = {token: i for i, token in enumerate(self.tokens)}
        return tuple(
            index.get(word, UNK_TOKEN) for word in mission.lower().split()
        )

    def vocab_hash(self) -> str:
        return hash_json(list(self.tokens))


def build_vocabulary(missions: list[str]) -> Vocabulary:
    """Sorted unique-token construction: independent of mission order."""
    words: set[str] = set()
    for mission in missions:
        words.update(mission.lower().split())
    return Vocabulary(tokens=("<pad>", "<unk>", *sorted(words)))


def vocabulary_from_dataset(dataset_dir: Path) -> Vocabulary:
    """Build the vocabulary from the stored dataset's mission sidecars only."""
    index_rows = read_jsonl(Path(dataset_dir) / "episode_index.jsonl")
    missions = []
    for row in index_rows:
        _, sidecar = read_episode(Path(dataset_dir) / "episodes", row["episode_id"])
        missions.append(sidecar.mission)
    return build_vocabulary(missions)


# --- Target windows ----------------------------------------------------------

@dataclass(frozen=True)
class WindowSpec:
    """One training item: a contiguous history window ending at one target."""

    window_id: str
    episode_id: str
    target_t: int
    start_t: int

    def source_step_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{self.episode_id}:{t}" for t in range(self.start_t, self.target_t + 1)
        )

    @property
    def length(self) -> int:
        return self.target_t - self.start_t + 1


@dataclass(frozen=True)
class TargetWindow:
    """Materialized window tensors (as NumPy arrays; padding happens in collation)."""

    spec: WindowSpec
    images: np.ndarray            # [T,7,7,3] uint8
    direction: np.ndarray         # [T] uint8
    prev_action_tokens: np.ndarray  # [T] int64, START token only at absolute t=0
    mission_tokens: np.ndarray    # [L] int64
    target_action: int
    checksum: str


def start_action_token(num_actions: int) -> int:
    """The model-level START token id; storage NULL_ACTION never leaves storage."""
    return num_actions


def enumerate_windows(
    episode_id: str,
    arrays: EpisodeArrays,
    max_context_prefix: int,
    max_sequence_length: int,
) -> list[WindowSpec]:
    """One window per revealed target, ending exactly at that target."""
    specs: list[WindowSpec] = []
    for target_t in np.flatnonzero(arrays.target_revealed):
        target_t = int(target_t)
        start_t = max(0, target_t - max_context_prefix)
        spec = WindowSpec(
            window_id=f"{episode_id}:{target_t}",
            episode_id=episode_id,
            target_t=target_t,
            start_t=start_t,
        )
        if spec.length > max_sequence_length:
            raise ManifestError(
                f"window {spec.window_id} has length {spec.length} beyond the cap "
                f"{max_sequence_length}"
            )
        specs.append(spec)
    return specs


def window_checksum(
    spec: WindowSpec, images, direction, prev_tokens, mission_tokens, target_action: int
) -> str:
    digest = hashlib.sha256()
    digest.update(spec.window_id.encode("utf-8") + b"\x1f")
    digest.update(str(spec.start_t).encode("ascii") + b"\x1f")
    for array in (images, direction, prev_tokens, mission_tokens):
        digest.update(np.ascontiguousarray(array).tobytes() + b"\x1f")
    digest.update(str(int(target_action)).encode("ascii"))
    return digest.hexdigest()


def materialize_window(
    arrays: EpisodeArrays,
    sidecar: EpisodeSidecar,
    spec: WindowSpec,
    vocab: Vocabulary,
    num_actions: int,
) -> TargetWindow:
    """Slice one window out of stored arrays and map storage sentinels to tokens."""
    if not bool(arrays.target_revealed[spec.target_t]):
        raise ManifestError(f"window {spec.window_id}: target step is not revealed")
    sl = slice(spec.start_t, spec.target_t + 1)
    images = np.ascontiguousarray(arrays.images[sl])
    direction = np.ascontiguousarray(arrays.direction[sl])
    raw_prev = arrays.previous_executed_action[sl].astype(np.int64)
    start_token = start_action_token(num_actions)
    prev_tokens = raw_prev.copy()
    null_positions = np.flatnonzero(raw_prev == NULL_ACTION)
    # The storage null appears exactly at absolute t=0; a clipped mid-episode
    # window must begin with the true executed action, never a fabricated START.
    for position in null_positions:
        if spec.start_t + int(position) != 0:
            raise ManifestError(
                f"window {spec.window_id}: null previous action inside the episode"
            )
        prev_tokens[position] = start_token
    mission_tokens = np.asarray(vocab.encode(sidecar.mission), dtype=np.int64)
    if mission_tokens.size == 0:
        raise ManifestError(f"window {spec.window_id}: empty mission token sequence")
    target_action = int(arrays.oracle_recommended_action[spec.target_t])
    if not 0 <= target_action < num_actions:
        raise ManifestError(
            f"window {spec.window_id}: target action {target_action} outside the action set"
        )
    checksum = window_checksum(
        spec, images, direction, prev_tokens, mission_tokens, target_action
    )
    return TargetWindow(
        spec=spec,
        images=images,
        direction=direction,
        prev_action_tokens=prev_tokens,
        mission_tokens=mission_tokens,
        target_action=target_action,
        checksum=checksum,
    )


@dataclass(frozen=True)
class Batch:
    """The only object the model may receive; padding exists only here."""

    image: torch.Tensor               # [B,T,7,7,3] long
    direction: torch.Tensor           # [B,T] long
    prev_executed_action: torch.Tensor  # [B,T] long
    mission_tokens: torch.Tensor      # [B,L] long, PAD-padded
    mission_lengths: torch.Tensor     # [B] long
    step_mask: torch.Tensor           # [B,T] bool
    target_mask: torch.Tensor         # [B,T] bool, exactly one True per row
    targets: torch.Tensor             # [B,T] long, valid only where target_mask


def collate_windows(
    windows: list[TargetWindow],
    channel_limits: tuple[int, int, int] | None = None,
) -> Batch:
    """Right-pad windows into one batch; the single place padding may occur."""
    # torch is imported here, not at module top: the manifest/preflight paths
    # of this module must stay importable without paying torch's startup cost.
    import torch

    if not windows:
        raise ManifestError("cannot collate an empty batch")
    batch_size = len(windows)
    max_t = max(window.spec.length for window in windows)
    max_l = max(int(window.mission_tokens.shape[0]) for window in windows)

    image = torch.zeros((batch_size, max_t, 7, 7, 3), dtype=torch.long)
    direction = torch.zeros((batch_size, max_t), dtype=torch.long)
    prev_action = torch.zeros((batch_size, max_t), dtype=torch.long)
    mission = torch.full((batch_size, max_l), PAD_TOKEN, dtype=torch.long)
    mission_lengths = torch.zeros(batch_size, dtype=torch.long)
    step_mask = torch.zeros((batch_size, max_t), dtype=torch.bool)
    target_mask = torch.zeros((batch_size, max_t), dtype=torch.bool)
    targets = torch.zeros((batch_size, max_t), dtype=torch.long)

    for row, window in enumerate(windows):
        length = window.spec.length
        image_np = np.asarray(window.images, dtype=np.int64)
        if channel_limits is not None:
            for channel, limit in enumerate(channel_limits):
                observed = int(image_np[..., channel].max())
                if observed >= limit:
                    raise ManifestError(
                        f"window {window.spec.window_id}: image channel {channel} value "
                        f"{observed} exceeds the configured vocabulary size {limit}"
                    )
        image[row, :length] = torch.from_numpy(image_np)
        direction[row, :length] = torch.from_numpy(
            np.asarray(window.direction, dtype=np.int64)
        )
        prev_action[row, :length] = torch.from_numpy(
            np.asarray(window.prev_action_tokens, dtype=np.int64)
        )
        mission_len = int(window.mission_tokens.shape[0])
        mission[row, :mission_len] = torch.from_numpy(
            np.asarray(window.mission_tokens, dtype=np.int64)
        )
        mission_lengths[row] = mission_len
        step_mask[row, :length] = True
        target_mask[row, length - 1] = True
        targets[row, length - 1] = window.target_action

    return Batch(
        image=image,
        direction=direction,
        prev_executed_action=prev_action,
        mission_tokens=mission,
        mission_lengths=mission_lengths,
        step_mask=step_mask,
        target_mask=target_mask,
        targets=targets,
    )


# --- Recovery reveal window --------------------------------------------------

def reveal_window_mask(
    episode_length: int, scheduled_time: int, h: int, budget_remaining: int
) -> np.ndarray:
    """Reveal mask for one recovery episode.

    The first recovery label is the oracle recommendation at the first state
    after the corrupted transition (``scheduled_time + 1``); at most ``h``
    successive active states are revealed, further capped by the remaining
    round budget. An episode that ends before or at the corruption reveals
    nothing; an episode that ends inside the window contributes only the
    labels actually reached.
    """
    if budget_remaining < 0:
        raise ManifestError("budget_remaining must be non-negative")
    mask = np.zeros(episode_length, dtype=np.bool_)
    first = scheduled_time + 1
    if first >= episode_length:
        return mask
    last = min(first + h - 1, episode_length - 1, first + budget_remaining - 1)
    if budget_remaining == 0:
        return mask
    mask[first : last + 1] = True
    return mask


# --- Eligible unseen test panel ----------------------------------------------

ELIGIBLE_FILE = "test_eligible.json"


def derive_eligible_subset(cfg: ExperimentConfig, manifest_root: Path) -> dict[str, object]:
    """Derive and store the eligible unseen test panel (preregistered rule).

    Eligibility uses only the nominal oracle path length recorded in the
    frozen test-candidate manifest: a scenario is eligible iff its nominal
    path is strictly longer than the latest primary intervention time, so the
    corruption is deliverable in principle. No learner output is involved.
    """
    if cfg.perturbation.unseen_time_set is None:
        raise ManifestError("unseen_time_set must be resolved before deriving eligibility")
    latest = max(cfg.perturbation.unseen_time_set)
    entries, candidate_hash = load_split_manifest(manifest_root, "test_candidate")
    eligible = [
        entry.ordinal
        for entry in entries
        if entry.nominal_oracle_path_length > latest
    ]
    payload = {
        "rule": "eligible iff nominal_oracle_path_length > latest_primary_intervention_time",
        "latest_primary_intervention_time": latest,
        "candidate_manifest_hash": candidate_hash,
        "candidate_count": len(entries),
        "eligible_count": len(eligible),
        "retained_fraction": len(eligible) / len(entries),
        "eligible_ordinals": eligible,
        "scenario_identity_hash": scenario_identity_hash(cfg),
    }
    payload["eligible_hash"] = hash_json(
        {"candidate_manifest_hash": candidate_hash, "eligible_ordinals": eligible}
    )
    atomic_write_json(Path(manifest_root) / ELIGIBLE_FILE, payload)
    return payload


def load_eligible_entries(
    cfg: ExperimentConfig, manifest_root: Path
) -> tuple[list[ManifestEntry], dict[str, object]]:
    """Load the frozen eligible panel, re-verifying its derivation."""
    payload = read_json(Path(manifest_root) / ELIGIBLE_FILE)
    entries, candidate_hash = load_split_manifest(manifest_root, "test_candidate")
    if payload["candidate_manifest_hash"] != candidate_hash:
        raise ManifestError("eligible subset was derived from a different candidate manifest")
    if payload["scenario_identity_hash"] != scenario_identity_hash(cfg):
        raise ManifestError("eligible subset was derived under a different scenario identity")
    expected_hash = hash_json(
        {
            "candidate_manifest_hash": candidate_hash,
            "eligible_ordinals": payload["eligible_ordinals"],
        }
    )
    if payload["eligible_hash"] != expected_hash:
        raise ManifestError("eligible subset hash does not verify")
    if cfg.perturbation.unseen_time_set is not None:
        latest = max(cfg.perturbation.unseen_time_set)
        if payload["latest_primary_intervention_time"] != latest:
            raise ManifestError(
                "eligible subset was derived for a different unseen schedule"
            )
    ordinals = set(payload["eligible_ordinals"])
    eligible_entries = [entry for entry in entries if entry.ordinal in ordinals]
    return eligible_entries, payload
