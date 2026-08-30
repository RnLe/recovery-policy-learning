// Step through a real episode: the full maze on the left (with the agent's
// field of view highlighted), exactly what the policy observes on the right.
// Both panes are drawn from the same symbolic data the models consume; no
// screenshots involved.

import { publicUrl } from "../data/paths";

type TrajectoryStep = {
  t: number;
  grid: number[][][];
  agent: { x: number; y: number; dir: number };
  observation: number[][][];
  visible: Array<[number, number]>;
  action: number | null; // null on the terminal state
  label: number | null;
  corrupted: boolean;
};

type Trajectory = {
  mission: string;
  outcome: "success" | "failure";
  steps_taken: number;
  corruption_time: number | null;
  action_names: string[];
  legend: { objects: Record<string, string>; colors: Record<string, string> };
  steps: TrajectoryStep[];
};

// MiniGrid's own object colors, so the drawing matches the rendered videos.
const OBJECT_COLORS: Record<string, string> = {
  red: "#e04040",
  green: "#40c040",
  blue: "#4066e0",
  purple: "#7027c3",
  yellow: "#d8c030",
  grey: "#9a9a9a",
};

const FLOOR = "#1c1c1c";
const WALL = "#5a5a5a";
const HIGHLIGHT = "rgba(255, 255, 255, 0.18)";
const AGENT = "#e63946";

function drawCell(
  context: CanvasRenderingContext2D,
  cell: [number, number, number],
  x: number,
  y: number,
  size: number,
  objects: Record<string, string>,
  colors: Record<string, string>,
): void {
  const [objectIndex, colorIndex, state] = cell;
  const objectName = objects[String(objectIndex)] ?? "unseen";
  const colorName = colors[String(colorIndex)] ?? "grey";
  const tint = OBJECT_COLORS[colorName] ?? "#9a9a9a";
  const px = x * size;
  const py = y * size;

  context.fillStyle = objectName === "unseen" ? "#000" : FLOOR;
  context.fillRect(px, py, size, size);

  switch (objectName) {
    case "wall":
      context.fillStyle = WALL;
      context.fillRect(px, py, size, size);
      break;
    case "door": {
      context.strokeStyle = tint;
      context.lineWidth = Math.max(2, size / 8);
      if (state === 0) {
        // open: door leaf against the frame
        context.strokeRect(px + 1, py + 1, size / 4, size - 2);
      } else {
        context.strokeRect(px + 2, py + 2, size - 4, size - 4);
      }
      break;
    }
    case "ball":
      context.fillStyle = tint;
      context.beginPath();
      context.arc(px + size / 2, py + size / 2, size * 0.32, 0, Math.PI * 2);
      context.fill();
      break;
    case "box":
      context.strokeStyle = tint;
      context.lineWidth = Math.max(2, size / 7);
      context.strokeRect(px + size * 0.18, py + size * 0.18, size * 0.64, size * 0.64);
      break;
    case "key": {
      context.strokeStyle = tint;
      context.fillStyle = tint;
      context.lineWidth = Math.max(2, size / 9);
      context.beginPath();
      context.arc(px + size * 0.42, py + size * 0.34, size * 0.14, 0, Math.PI * 2);
      context.stroke();
      context.fillRect(px + size * 0.52, py + size * 0.42, size * 0.11, size * 0.36);
      break;
    }
    case "goal":
      context.fillStyle = "#40c040";
      context.fillRect(px + 1, py + 1, size - 2, size - 2);
      break;
    default:
      break;
  }
}

function drawAgent(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  direction: number,
  size: number,
): void {
  const cx = x * size + size / 2;
  const cy = y * size + size / 2;
  context.save();
  context.translate(cx, cy);
  context.rotate((direction * Math.PI) / 2); // 0 faces east
  context.fillStyle = AGENT;
  context.beginPath();
  context.moveTo(size * 0.34, 0);
  context.lineTo(-size * 0.26, -size * 0.26);
  context.lineTo(-size * 0.26, size * 0.26);
  context.closePath();
  context.fill();
  context.restore();
}

export async function mountScrubbers(): Promise<void> {
  const reducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;

  for (const slot of document.querySelectorAll<HTMLElement>("[data-scrubber]")) {
    const trace = slot.dataset.scrubber ?? "";
    let trajectory: Trajectory;
    try {
      const response = await fetch(
        publicUrl(`media/journey/trajectories/${trace}.json`),
      );
      if (!response.ok) throw new Error(String(response.status));
      trajectory = (await response.json()) as Trajectory;
    } catch {
      slot.innerHTML =
        '<p class="muted small">trajectory data unavailable: run the staging step</p>';
      continue;
    }

    const cellSize = 46; // backing store well above display size, so zoom stays crisp
    const gridWidth = trajectory.steps[0]!.grid.length;
    const gridHeight = trajectory.steps[0]!.grid[0]!.length;

    slot.className = "scrubber";
    slot.innerHTML = `
      <p>“${trajectory.mission}”,
        <span class="muted">${
          trajectory.outcome === "success"
            ? `solved in ${trajectory.steps_taken} steps`
            : `${trajectory.steps_taken} steps, and the goal stays out of reach`
        }</span></p>
      <div class="panes">
        <div>
          <p class="small muted">the world (the agent never sees this; the
            bright cells are its field of view)</p>
          <canvas data-pane="world" width="${gridWidth * cellSize}"
                  height="${gridHeight * cellSize}"
                  role="img" aria-label="full maze at the current step"></canvas>
        </div>
        <div>
          <p class="small muted">what the policy observes (7×7 symbolic crop,
            agent at the bottom center facing up)</p>
          <canvas data-pane="observation" width="${7 * 46}" height="${7 * 46}"
                  role="img" aria-label="the agent's observation at the current step"></canvas>
        </div>
      </div>
      <div class="controls">
        <button type="button" data-ctl="back" aria-label="previous step">‹</button>
        <button type="button" data-ctl="play">play</button>
        <button type="button" data-ctl="next" aria-label="next step">›</button>
        <input type="range" min="0" max="${trajectory.steps.length - 1}" value="0"
               step="1" aria-label="episode step">
      </div>
      <p class="readout" aria-live="polite"></p>
    `;

    const world = slot.querySelector<HTMLCanvasElement>('[data-pane="world"]')!;
    const observed = slot.querySelector<HTMLCanvasElement>(
      '[data-pane="observation"]',
    )!;
    const slider = slot.querySelector<HTMLInputElement>("input[type=range]")!;
    const readout = slot.querySelector<HTMLElement>(".readout")!;
    const playButton = slot.querySelector<HTMLButtonElement>('[data-ctl="play"]')!;
    const worldContext = world.getContext("2d")!;
    const observationContext = observed.getContext("2d")!;
    const { objects, colors } = trajectory.legend;

    const render = (index: number) => {
      const step = trajectory.steps[index]!;
      for (let x = 0; x < gridWidth; x += 1) {
        for (let y = 0; y < gridHeight; y += 1) {
          drawCell(
            worldContext,
            step.grid[x]![y]! as [number, number, number],
            x, y, cellSize, objects, colors,
          );
        }
      }
      worldContext.fillStyle = HIGHLIGHT;
      for (const [x, y] of step.visible) {
        worldContext.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
      }
      drawAgent(worldContext, step.agent.x, step.agent.y, step.agent.dir, cellSize);
      if (step.corrupted) {
        worldContext.strokeStyle = "#eba538";
        worldContext.lineWidth = 8;
        worldContext.strokeRect(0, 0, world.width, world.height);
      }

      const obsCell = 46;
      for (let x = 0; x < 7; x += 1) {
        for (let y = 0; y < 7; y += 1) {
          drawCell(
            observationContext,
            step.observation[x]![y]! as [number, number, number],
            x, y, obsCell, objects, colors,
          );
        }
      }
      drawAgent(observationContext, 3, 6, 3, obsCell); // bottom center, facing up

      const labelName =
        step.label !== null ? trajectory.action_names[step.label] : null;
      if (step.action === null) {
        readout.innerHTML =
          `Step ${step.t}, where the episode ends in ` +
          `<b>${trajectory.outcome}</b>.`;
      } else {
        const actionName = trajectory.action_names[step.action] ?? "?";
        const agreeing = labelName !== null && labelName === actionName;
        readout.innerHTML =
          `Step ${step.t}: the agent goes <b>${actionName}</b>` +
          (labelName
            ? agreeing
              ? ", which is what the oracle labels here."
              : `, while the oracle labels this state <b>${labelName}</b>.`
            : ".") +
          (step.corrupted
            ? ' <span class="flash">⚡ this action was corrupted</span>'
            : "");
      }
    };

    let timer: number | null = null;
    const stop = () => {
      if (timer !== null) window.clearInterval(timer);
      timer = null;
      playButton.textContent = "play";
    };
    const go = (index: number) => {
      const clamped = Math.max(0, Math.min(trajectory.steps.length - 1, index));
      slider.value = String(clamped);
      render(clamped);
    };
    slider.addEventListener("input", () => {
      stop();
      render(Number(slider.value));
    });
    slot.querySelector('[data-ctl="back"]')!.addEventListener("click", () => {
      stop();
      go(Number(slider.value) - 1);
    });
    slot.querySelector('[data-ctl="next"]')!.addEventListener("click", () => {
      stop();
      go(Number(slider.value) + 1);
    });
    if (reducedMotion) {
      playButton.hidden = true;
    } else {
      playButton.addEventListener("click", () => {
        if (timer !== null) {
          stop();
          return;
        }
        playButton.textContent = "pause";
        timer = window.setInterval(() => {
          const next = Number(slider.value) + 1;
          if (next >= trajectory.steps.length) {
            stop();
            return;
          }
          go(next);
        }, 400);
      });
    }
    render(0);
  }
}
