"""Shared plumbing for the foundations labs.

The labs form a deterministic, tested teaching track beside the frozen study.
This module provides the pieces every lab needs: named seeds in a namespace
disjoint from the study's, a fixed output layout, exploratory-evidence
labelling, and writers for metrics, tables, figures, mini-reports, and
generated Typst fragments, so that each ``labNN`` module contains only the
concept it teaches.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# One plot style for the whole track: dark-steel chrome on white, generous
# text (figures are read at column width), and deterministic SVG output so
# staged bytes are stable across reruns.
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#4f5f6b",
        "axes.linewidth": 1.5,
        "text.color": "#4f5f6b",
        "axes.labelcolor": "#4f5f6b",
        "xtick.color": "#4f5f6b",
        "ytick.color": "#4f5f6b",
        "svg.fonttype": "path",
        "svg.hashsalt": "gr-foundations",
    }
)

import matplotlib.pyplot as plt  # noqa: E402

from grounded_recovery.artifacts import (  # noqa: E402
    atomic_write_json,
    hash_json,
)

# Domain-separated seeding: the tag guarantees that no foundations seed can
# collide with a study seed stream even for equal component names.
FOUNDATIONS_DOMAIN = "gr-foundations"
FOUNDATIONS_ROOT_SEED = 20260826

EXPLORATORY_LABEL = (
    "EXPLORATORY, FOUNDATIONS (pedagogical companion; not confirmatory evidence)"
)

# Shared plot vocabulary, the same arm colors as every figure on the site:
# sage for recovery, sky blue for extra demonstrations, dusty brown for the
# base policy, stark orange for corruptions.
COLOR_BASE = "#857255"
COLOR_EXTRA = "#4e9ae1"
COLOR_RECOVERY = "#6b8f71"
COLOR_CAUTION = "#eba538"
COLOR_NEUTRAL = "#4f5f6b"
COLOR_DANGER = "#c0392b"

SEED_COMPONENTS = frozenset(
    {
        "lab01.census",
        "lab01.gallery",
        "lab02.rollouts",
        "lab03.random_policy",
        "lab03.oracle_eval",
        "lab03.sync_experiment",
        "lab03.trajectory",
        "lab04.qlearning",
        "lab04.dataset",
        "lab04.holdout",
        "lab04.train",
        "lab05.train",
        "lab06.sweep",
        "lab06.dataset",
        "lab06.holdout",
        "lab06.train",
        "lab06.collection",
        "lab07.simulation",
        "lab07.power",
        "lab07.tamper",
        "lab07.unmatched",
    }
)

LAB_TITLES: dict[str, str] = {
    "lab01": "The world: BabyAI/MiniGrid",
    "lab02": "Decision processes: from MDP to POMDP",
    "lab03": "Policies, oracles, and where labels come from",
    "lab04": "Learning paradigms: reinforcement versus imitation",
    "lab05": "The policy network, piece by piece",
    "lab06": "When cloning breaks: shift, corruptions, recovery",
    "lab07": "Measuring honestly: budgets, ITT, and frozen protocols",
}


class FoundationsError(RuntimeError):
    """A foundations lab was driven outside its contract."""


def derive_seed(component: str, index: int = 0) -> int:
    """Deterministic 63-bit seed for a named foundations component."""
    if component not in SEED_COMPONENTS:
        raise FoundationsError(
            f"unknown foundations seed component {component!r}; "
            "register it in SEED_COMPONENTS"
        )
    if index < 0:
        raise FoundationsError(f"seed index must be non-negative, got {index}")
    payload = "\x1f".join(
        [FOUNDATIONS_DOMAIN, str(FOUNDATIONS_ROOT_SEED), component, str(index)]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class LabPaths:
    """Fixed output layout for one lab."""

    lab_id: str
    repo_root: Path

    def __post_init__(self) -> None:
        if self.lab_id not in LAB_TITLES:
            raise FoundationsError(f"unknown lab id {self.lab_id!r}")

    @property
    def title(self) -> str:
        return LAB_TITLES[self.lab_id]

    @property
    def data_dir(self) -> Path:
        """Heavy artifacts (datasets, checkpoints); gitignored via data/."""
        return self.repo_root / "data" / "foundations" / self.lab_id

    @property
    def out_dir(self) -> Path:
        """Small committable outputs: metrics, tables, figures, mini-report."""
        return self.repo_root / "foundations" / self.lab_id

    @property
    def figures_dir(self) -> Path:
        return self.out_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.out_dir / "tables"

    @property
    def report_dir(self) -> Path:
        """Generated Typst fragments and report figure copies."""
        return self.repo_root / "report" / "generated" / "foundations" / self.lab_id


def prepare(paths: LabPaths, *, force: bool) -> None:
    """Create the lab's output directories; refuse to clobber without force."""
    existing = [d for d in (paths.out_dir, paths.report_dir, paths.data_dir) if d.exists()]
    if existing and not force:
        listing = ", ".join(str(d) for d in existing)
        raise FoundationsError(
            f"{paths.lab_id} outputs already exist ({listing}); "
            "pass --force to rerun and replace them"
        )
    for directory in existing:
        shutil.rmtree(directory)
    for directory in (
        paths.data_dir,
        paths.figures_dir,
        paths.tables_dir,
        paths.report_dir / "figures",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def write_metrics(paths: LabPaths, metrics: dict[str, object]) -> str:
    """Write the lab's deterministic metrics document; returns its content hash.

    Timestamps and durations belong in ``run_info.json`` (written by the CLI),
    never here: metrics.json must be byte-identical across reruns.
    """
    document = {
        "lab": paths.lab_id,
        "title": paths.title,
        "evidence_label": EXPLORATORY_LABEL,
        "root_seed": FOUNDATIONS_ROOT_SEED,
        "seed_domain": FOUNDATIONS_DOMAIN,
        "metrics": metrics,
    }
    atomic_write_json(paths.out_dir / "metrics.json", document)
    return hash_json(document)


def write_mini_report(
    paths: LabPaths,
    *,
    question: str,
    sections: list[tuple[str, str]],
) -> Path:
    """Write the lab's human-readable mini-report."""
    lines = [
        f"# Foundations {paths.lab_id}: {paths.title}",
        "",
        f"**Status:** {EXPLORATORY_LABEL}",
        "",
        f"**Question answered:** {question}",
        "",
        f"Reproduce with `uv run grf run {paths.lab_id} --force` "
        "(deterministic; named seeds under the gr-foundations domain).",
        "",
    ]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", body.rstrip(), ""])
    path = paths.out_dir / "mini_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _typst_header(paths: LabPaths) -> str:
    return (
        f"// GENERATED. DO NOT EDIT. (gr_foundations {paths.lab_id})\n"
        f"// Evidence status: exploratory foundations material, not confirmatory.\n"
    )


def _typst_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def export_typst_values(paths: LabPaths, name: str, values: dict[str, str]) -> Path:
    """Export ``#let`` bindings for use in the lab's report chapter."""
    lines = [_typst_header(paths)]
    for key, value in values.items():
        if not key.replace("-", "").isalnum():
            raise FoundationsError(f"invalid Typst identifier {key!r}")
        lines.append(f'#let {key} = "{_typst_escape(str(value))}"')
    path = paths.report_dir / f"{name}.typ"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def export_typst_table(
    paths: LabPaths,
    name: str,
    header: list[str],
    rows: list[list[object]],
) -> Path:
    """Export a simple Typst table fragment."""
    for row in rows:
        if len(row) != len(header):
            raise FoundationsError(
                f"table {name!r}: row width {len(row)} != header width {len(header)}"
            )
    cells = [f"[*{_typst_escape(cell)}*]" for cell in header]
    for row in rows:
        cells.extend(f"[{_typst_escape(str(cell))}]" for cell in row)
    body = ",\n  ".join(cells)
    content = (
        f"{_typst_header(paths)}"
        f"#table(\n  columns: {len(header)},\n  {body},\n)\n"
    )
    path = paths.report_dir / f"{name}.typ"
    path.write_text(content, encoding="utf-8")
    return path


# Glyphs rendered as outline paths must not inherit a stroke, or Typst's SVG
# renderer fattens the narrow ones; only the scaled glyph defs are touched.
_GLYPH_PATH = re.compile(
    r'(<path id="[^"]+" d="[^"]*" transform="scale\([^"]+\)")\s*/>'
)


def save_figure(paths: LabPaths, figure: plt.Figure, name: str) -> Path:
    """Save a figure to the lab outputs and mirror it into the report assets."""
    if "/" in name or not name.endswith(".svg"):
        raise FoundationsError(f"figure name must be a bare *.svg name, got {name!r}")
    out_path = paths.figures_dir / name
    figure.savefig(out_path, bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    text = out_path.read_text(encoding="utf-8")
    out_path.write_text(_GLYPH_PATH.sub(r'\1 stroke="none"/>', text), encoding="utf-8")
    shutil.copyfile(out_path, paths.report_dir / "figures" / name)
    return out_path


def write_table_csv(
    paths: LabPaths, name: str, header: list[str], rows: list[list[object]]
) -> Path:
    """Write a small CSV table into the lab outputs."""
    import csv

    if "/" in name or not name.endswith(".csv"):
        raise FoundationsError(f"table name must be a bare *.csv name, got {name!r}")
    path = paths.tables_dir / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path
