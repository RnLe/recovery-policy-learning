"""Vector rendering of grid states for the foundations figures.

Figures used to embed raster screenshots from the environment's renderer;
those pixels survive any export as pixels. Everything here draws the same
symbolic grid the models actually consume (walls, doors, objects, the agent
triangle, and its field of view) as matplotlib patches, so a figure saved as
SVG stays sharp at any zoom. The cell shapes and colors mirror the site's
canvas scrubber, which keeps worlds recognisable across figures, videos, and
the in-page step-through.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Polygon, Rectangle
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT

from grounded_recovery.world import WorldSession

# MiniGrid's own object colors, matching the rendered videos and the scrubber.
OBJECT_TINTS = {
    "red": "#e04040",
    "green": "#40c040",
    "blue": "#4066e0",
    "purple": "#7027c3",
    "yellow": "#d8c030",
    "grey": "#9a9a9a",
}
UNSEEN = "#000000"
FLOOR = "#1c1c1c"
WALL = "#5a5a5a"
CELL_EDGE = "#2e2e2e"
AGENT = "#e63946"

# The site's value ramp (dark blue → light orange), for integer-plane heatmaps.
VALUE_RAMP = ["#4f5f6b", "#4d7b9e", "#a5c6df", "#f7f4e9", "#e3b064"]
value_colormap = LinearSegmentedColormap.from_list("gr-value", VALUE_RAMP, N=256)


def state_snapshot(
    session: WorldSession,
) -> tuple[np.ndarray, tuple[int, int, int], list[tuple[int, int]]]:
    """Grid encoding, agent pose, and the currently visible cells."""
    unwrapped = session.env.unwrapped
    grid = np.asarray(unwrapped.grid.encode(), dtype=np.int16)
    pose = (
        int(unwrapped.agent_pos[0]),
        int(unwrapped.agent_pos[1]),
        int(unwrapped.agent_dir),
    )
    # Straight from the environment's own visibility test.
    visible = [
        (x, y)
        for x in range(grid.shape[0])
        for y in range(grid.shape[1])
        if unwrapped.in_view(x, y)
    ]
    return grid, pose, visible


def _draw_cell(axis: Axes, x: int, y: int, cell: np.ndarray) -> None:
    obj_index, color_index, state = (int(v) for v in cell)
    name = IDX_TO_OBJECT.get(obj_index, "unseen")
    tint = OBJECT_TINTS.get(IDX_TO_COLOR.get(color_index, "grey"), "#9a9a9a")

    axis.add_patch(
        Rectangle(
            (x, y), 1, 1,
            facecolor=UNSEEN if name == "unseen" else FLOOR,
            edgecolor=CELL_EDGE, linewidth=0.4,
        )
    )
    if name == "wall":
        axis.add_patch(
            Rectangle((x, y), 1, 1, facecolor=WALL, edgecolor=CELL_EDGE, linewidth=0.4)
        )
    elif name == "door":
        if state == 0:  # open: the leaf rests against the frame
            axis.add_patch(
                Rectangle(
                    (x + 0.04, y + 0.04), 0.24, 0.92,
                    facecolor="none", edgecolor=tint, linewidth=1.6,
                )
            )
        else:
            axis.add_patch(
                Rectangle(
                    (x + 0.08, y + 0.08), 0.84, 0.84,
                    facecolor="none", edgecolor=tint, linewidth=1.6,
                )
            )
    elif name == "ball":
        axis.add_patch(Circle((x + 0.5, y + 0.5), 0.32, facecolor=tint))
    elif name == "box":
        axis.add_patch(
            Rectangle(
                (x + 0.18, y + 0.18), 0.64, 0.64,
                facecolor="none", edgecolor=tint, linewidth=1.8,
            )
        )
    elif name == "key":
        axis.add_patch(
            Circle(
                (x + 0.42, y + 0.34), 0.14,
                facecolor="none", edgecolor=tint, linewidth=1.4,
            )
        )
        axis.add_patch(Rectangle((x + 0.52, y + 0.42), 0.11, 0.36, facecolor=tint))
    elif name == "goal":
        axis.add_patch(
            Rectangle((x + 0.04, y + 0.04), 0.92, 0.92, facecolor="#40c040")
        )


def _draw_agent(axis: Axes, x: int, y: int, direction: int) -> None:
    # Same triangle as the scrubber, in the grid's y-down coordinates
    # (direction 0 faces east, then clockwise).
    angle = direction * np.pi / 2.0
    cos, sin = np.cos(angle), np.sin(angle)
    local = np.array([[0.34, 0.0], [-0.26, -0.26], [-0.26, 0.26]])
    rotated = local @ np.array([[cos, sin], [-sin, cos]])
    points = rotated + np.array([x + 0.5, y + 0.5])
    axis.add_patch(Polygon(points, closed=True, facecolor=AGENT))


def draw_grid(
    axis: Axes,
    grid: np.ndarray,
    *,
    agent: tuple[int, int, int] | None = None,
    visible: list[tuple[int, int]] | None = None,
) -> None:
    """Draw one full world state onto an axis as pure vector patches."""
    width, height = int(grid.shape[0]), int(grid.shape[1])
    for x in range(width):
        for y in range(height):
            _draw_cell(axis, x, y, grid[x, y])
    if visible:
        for x, y in visible:
            axis.add_patch(
                Rectangle((x, y), 1, 1, facecolor="white", alpha=0.18, edgecolor="none")
            )
    if agent is not None:
        _draw_agent(axis, agent[0], agent[1], agent[2])
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)  # y grows downward, like the environment's render
    axis.set_aspect("equal")
    axis.axis("off")


@dataclass(frozen=True)
class WorldState:
    """One drawable world state, detached from its session."""

    grid: np.ndarray
    agent: tuple[int, int, int]
    visible: list[tuple[int, int]]


def draw_world(axis: Axes, session: WorldSession) -> None:
    """Draw the session's current state with its field of view highlighted."""
    grid, pose, visible = state_snapshot(session)
    draw_grid(axis, grid, agent=pose, visible=visible)


def draw_state(axis: Axes, state: WorldState) -> None:
    """Draw a previously captured world state."""
    draw_grid(axis, state.grid, agent=state.agent, visible=state.visible)


def draw_plane(
    axis: Axes,
    plane: np.ndarray,
    *,
    vmax: float,
    glyphs: list[list[str]] | None = None,
) -> None:
    """Draw a small integer plane as a vector heatmap on the value ramp."""
    rows, columns = plane.shape
    for row in range(rows):
        for column in range(columns):
            value = float(plane[row, column])
            axis.add_patch(
                Rectangle(
                    (column, row), 1, 1,
                    facecolor=value_colormap(value / vmax if vmax else 0.0),
                    edgecolor="white", linewidth=0.6,
                )
            )
            if glyphs is not None:
                shade = value / vmax if vmax else 0.0
                axis.text(
                    column + 0.5, row + 0.5, glyphs[row][column],
                    ha="center", va="center", fontsize=11,
                    color="white" if shade < 0.55 else "#4f5f6b",
                )
    axis.set_xlim(0, columns)
    axis.set_ylim(rows, 0)
    axis.set_aspect("equal")
    axis.axis("off")
