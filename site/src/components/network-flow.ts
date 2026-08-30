// The policy network with its real numbers: every dot is one value from the
// staged trace (weights and activations of an actual trained checkpoint,
// replayed on a real episode), colored on the dark-blue → light-orange ramp.
// A slider scrubs the episode; play animates one forward pass per step.
// Colors are assigned in one synchronous pass per step; gsap only ever
// animates opacity and transforms on top of the recolored scene.

import { gsap, type Timeline } from "../anim/gsap";
import {
  prefersReducedMotion,
  svgEl,
  tokenColor,
  valueRamp,
} from "../anim/helpers";
import { publicUrl } from "../data/paths";

type Block = { shape: number[]; min: number; max: number; values: number[] };

type TraceStep = {
  t: number;
  obs: number[][][];
  direction: number;
  prev_action: number | null;
  acts: Record<string, Block>;
  action: number;
  oracle_label: number | null;
};

type Trace = {
  kind: string;
  variant: string;
  mission: { text: string; tokens: number[]; vocab: string[] };
  action_names: string[];
  outcome: string;
  steps_taken: number;
  exported_steps: number;
  weights: Record<string, Block | { tensor: Block }>;
  mission_acts: { per_token_hidden: Block; mission_vec: Block };
  ranges: Record<string, { min: number; max: number }>;
  steps: TraceStep[];
  source: Record<string, string | number>;
};

// Highest symbol index per observation plane (object, color, state).
const OBS_VMAX = [10, 5, 2];

type DotGrid = { dots: SVGCircleElement[]; rows: number; cols: number };

export async function mountNetworkFlows(): Promise<void> {
  for (const slot of document.querySelectorAll<HTMLElement>(
    "[data-network-flow]",
  )) {
    const id = slot.dataset.networkFlow ?? "full_r0";
    let trace: Trace;
    try {
      const response = await fetch(
        publicUrl(`media/journey/network/${id}.json`),
      );
      if (!response.ok) throw new Error(String(response.status));
      trace = (await response.json()) as Trace;
    } catch {
      slot.innerHTML =
        '<p class="muted small">network trace unavailable: run the staging step</p>';
      continue;
    }
    createNetworkFlow(slot, trace);
  }
}

export function createNetworkFlow(mount: HTMLElement, trace: Trace) {
  const reduced = prefersReducedMotion();
  const ramp = valueRamp([
    tokenColor("--value-0"),
    tokenColor("--value-1"),
    tokenColor("--value-2"),
    tokenColor("--value-3"),
    tokenColor("--value-4"),
  ]);
  const ink = "var(--ink-soft)";
  const line = "var(--line-strong)";
  const steps = trace.steps;
  const names = trace.action_names;

  mount.className = "scrubber netflow";
  mount.innerHTML = `
    <p>“${trace.mission.text}”,
      <span class="muted">the policy's own rollout, ${
        trace.outcome === "success" ? "solved" : "unsolved"
      } in ${trace.steps_taken} steps. Every dot is a real number from the
      trained checkpoint.</span></p>
    <svg class="netflow__stage" viewBox="0 0 1084 505" role="img">
      <title>The policy network, animated with its real values</title>
      <desc>Observation, mission, direction and previous-action encoders
        converge through a fusion layer and a recurrent core into three
        action logits. Dots are colored by the actual values.</desc>
    </svg>
    <div class="controls">
      <button type="button" data-ctl="back" aria-label="previous step">‹</button>
      <button type="button" data-ctl="play">play</button>
      <button type="button" data-ctl="next" aria-label="next step">›</button>
      <input type="range" min="0" max="${steps.length - 1}" value="0"
             step="1" aria-label="episode step">
    </div>
    <p class="readout" aria-live="polite"></p>
    <div class="netflow__legend">
      <span class="netflow__ramp" aria-hidden="true"></span>
      <span data-legend>each block is scaled to its own real range</span>
    </div>
  `;
  const svg = mount.querySelector<SVGSVGElement>("svg")!;
  const slider = mount.querySelector<HTMLInputElement>("input[type=range]")!;
  const readout = mount.querySelector<HTMLElement>(".readout")!;
  const legend = mount.querySelector<HTMLElement>("[data-legend]")!;
  const playButton = mount.querySelector<HTMLButtonElement>('[data-ctl="play"]')!;

  // ---- stage construction (everything created once) -----------------------

  const label = (x: number, y: number, text: string, size = 12) => {
    const el = svgEl("text", {
      x, y, fill: ink, "font-size": size, "font-family": "var(--font-body)",
    });
    el.textContent = text;
    svg.append(el);
    return el;
  };

  const inspectable = (
    el: SVGElement,
    name: string,
    block: { min: number; max: number },
    count: number,
  ) => {
    el.setAttribute("tabindex", "0");
    el.setAttribute("role", "img");
    el.setAttribute(
      "aria-label",
      `${name}: ${count} values, min ${block.min}, max ${block.max}`,
    );
    const show = () => {
      legend.textContent = `${name} · min ${block.min} · max ${block.max}`;
    };
    el.addEventListener("mouseenter", show);
    el.addEventListener("focus", show);
  };

  const dotGrid = (
    x: number,
    y: number,
    rows: number,
    cols: number,
    pitch: number,
    radius: number,
  ): DotGrid => {
    const group = svgEl("g");
    const dots: SVGCircleElement[] = [];
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const dot = svgEl("circle", {
          cx: x + c * pitch, cy: y + r * pitch, r: radius, fill: "#d8d2c4",
        });
        group.append(dot);
        dots.push(dot);
      }
    }
    svg.append(group);
    return { dots, rows, cols };
  };

  const box = (x: number, y: number, w: number, h: number, title: string) => {
    const rect = svgEl("rect", {
      x, y, width: w, height: h, rx: 8, fill: "var(--paper-raised)",
      stroke: line, "stroke-width": 1.5,
    });
    svg.append(rect);
    label(x + 10, y + 18, title, 12.5);
    return rect;
  };

  const connector = (d: string) => {
    const path = svgEl("path", {
      d, fill: "none", stroke: line, "stroke-width": 1.5,
    });
    svg.append(path);
    return path;
  };

  // Observation planes (transposed like every figure: row = depth of view).
  label(16, 18, "the observation · 7×7×3 symbols", 13);
  const planeNames = ["object", "color", "state"];
  const planes = planeNames.map((name, i) => {
    label(16 + i * 66, 34, name, 11);
    return dotGrid(22 + i * 66, 46, 7, 7, 8, 3.2);
  });

  // Static embedding tables under the planes.
  label(16, 116, "symbol embeddings (rows = symbols)", 11);
  const embedKeys = ["obs_embed_object", "obs_embed_color", "obs_embed_state"];
  embedKeys.forEach((key, i) => {
    const block = trace.weights[key] as Block;
    const grid = dotGrid(
      22 + i * 66, 128, block.shape[0]!, block.shape[1]!, 3.4, 1.3,
    );
    paint(grid.dots, block.values, block.min, block.max);
    inspectable(
      grid.dots[0]!.parentElement as unknown as SVGElement,
      `${planeNames[i]} embedding ${block.shape.join("×")}`,
      block, block.values.length,
    );
  });

  // Convolutions: sampled channels, with the sliding window over the planes.
  label(228, 18, "conv 3×3 ×2 (4 of 32 channels)", 13);
  const conv1 = dotGrid(238, 46, 10, 10, 7, 2.6); // 4 channels as 2×2 of 5×5
  const conv2 = dotGrid(330, 60, 6, 6, 7, 2.6); // 4 channels as 2×2 of 3×3
  const windowRect = svgEl("rect", {
    x: 22, y: 46 - 4, width: 8 * 2 + 8, height: 8 * 2 + 8, rx: 2,
    fill: "none", stroke: "var(--caution)", "stroke-width": 1.6, opacity: 0,
  });
  svg.append(windowRect);

  // Observation vector.
  label(420, 18, "observation vector · 64", 13);
  const obsVec = dotGrid(430, 46, 4, 16, 9, 3.4);

  // Mission lane.
  label(16, 216, "the mission, encoded once", 13);
  trace.mission.tokens.forEach((tokenId, i) => {
    const rect = svgEl("rect", {
      x: 16, y: 228 + i * 22, width: 74, height: 18, rx: 4,
      fill: "var(--paper-sunken)", stroke: line,
    });
    const text = svgEl("text", {
      x: 22, y: 241 + i * 22, fill: ink, "font-size": 10.5,
      "font-family": "var(--font-mono)",
    });
    text.textContent = trace.mission.vocab[tokenId] ?? "?";
    svg.append(rect, text);
  });
  label(110, 228, "word embedding 14×32", 11);
  const wordEmbedding = trace.weights["word_embedding"] as Block;
  const wordGrid = dotGrid(114, 240, 14, 32, 3.4, 1.3);
  paint(wordGrid.dots, wordEmbedding.values, wordEmbedding.min, wordEmbedding.max);
  inspectable(
    wordGrid.dots[0]!.parentElement as unknown as SVGElement,
    "word embedding 14×32", wordEmbedding, wordEmbedding.values.length,
  );
  connector("M 212 273 L 248 273");
  box(250, 250, 96, 46, "mission GRU");
  connector("M 346 273 C 380 273, 396 250, 428 246");
  const gruLoop = svgEl("path", {
    d: "M 316 250 C 322 234, 286 234, 292 250",
    fill: "none", stroke: line, "stroke-width": 1.4,
  });
  svg.append(gruLoop);
  label(420, 216, "mission vector · 64 · reused each step", 12);
  const missionVec = dotGrid(430, 232, 4, 16, 9, 3.4);

  // Direction and previous action.
  label(16, 366, "compass direction", 12);
  const dirStrip = dotGrid(150, 362, 1, 8, 11, 4);
  label(16, 404, "previous executed action", 12);
  const prevStrip = dotGrid(210, 400, 1, 8, 11, 4);
  const prevChip = svgEl("text", {
    x: 150, y: 404, fill: ink, "font-size": 11,
    "font-family": "var(--font-mono)",
  });
  svg.append(prevChip);

  // Trunk: concat → fusion → fused → GRU (with memory) → head.
  const concatX = 600;
  const pathObs = connector(`M 566 60 C ${concatX} 60, ${concatX} 150, 620 190`);
  const pathMission = connector(
    `M 566 246 C ${concatX} 246, ${concatX} 210, 620 210`,
  );
  const pathContext = connector(
    `M 300 384 C ${concatX} 384, ${concatX} 262, 620 228`,
  );
  box(620, 168, 118, 74, "fusion 144→128");
  const fusionBlock = trace.weights["fusion_block"] as Block;
  const fusionGrid = dotGrid(632, 192, 16, 16, 2.8, 1.1);
  paint(fusionGrid.dots, fusionBlock.values, fusionBlock.min, fusionBlock.max);
  inspectable(
    fusionGrid.dots[0]!.parentElement as unknown as SVGElement,
    "fusion weights (16×16 of 128×144)", fusionBlock, fusionBlock.values.length,
  );
  label(770, 160, "fused · 128", 12);
  const fusedVec = dotGrid(780, 176, 8, 16, 8, 3);
  connector("M 780 250 C 780 268, 790 276, 800 284");

  box(760, 290, 150, 96, "policy GRU · the memory");
  label(636, 300, "hidden from t−1", 10.5);
  const ghostVec = dotGrid(636, 312, 8, 16, 5, 1.8);
  connector("M 720 340 L 758 340");
  label(930, 284, "hidden · 128", 12);
  const hiddenVec = dotGrid(940, 300, 8, 16, 8, 3);
  const carryLoop = svgEl("path", {
    d: "M 990 372 C 990 430, 680 430, 676 356",
    fill: "none", stroke: line, "stroke-width": 1.4,
    "stroke-dasharray": "4 5",
  });
  svg.append(carryLoop);
  label(700, 452, "the hidden state carries to the next step", 11);

  // Head.
  connector("M 1000 292 C 1010 240, 1010 200, 1000 160");
  label(930, 84, "three logits → action", 13);
  const logitDots: SVGCircleElement[] = names.map((_, i) => {
    const dot = svgEl("circle", {
      cx: 950, cy: 108 + i * 30, r: 10, fill: "#d8d2c4",
    });
    svg.append(dot);
    return dot;
  });
  const logitLabels = names.map((name, i) =>
    label(968, 112 + i * 30, name, 12),
  );
  const chosenRing = svgEl("circle", {
    cx: 950, cy: 108, r: 15, fill: "none", stroke: "var(--ink-strong)",
    "stroke-width": 2.2,
  });
  svg.append(chosenRing);
  const oracleMark = svgEl("text", {
    x: 1030, y: 112, fill: "var(--caution-strong)", "font-size": 11,
  });
  svg.append(oracleMark);

  // Traveling pulses (motion mode only), one per converging connector.
  const pulses = [pathObs, pathMission, pathContext].map(() => {
    const pulse = svgEl("circle", {
      cx: 0, cy: 0, r: 4, fill: "var(--extra)", opacity: 0,
    });
    svg.append(pulse);
    return pulse;
  });

  // Legend ramp swatch (CSS gradient from the same tokens).
  const rampEl = mount.querySelector<HTMLElement>(".netflow__ramp")!;
  rampEl.style.background = `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1]
    .map((v) => ramp(v))
    .join(", ")})`;

  // ---- painting -----------------------------------------------------------

  function paint(
    dots: SVGCircleElement[],
    values: number[],
    min: number,
    max: number,
  ): void {
    const span = max - min || 1;
    const count = Math.min(dots.length, values.length);
    for (let i = 0; i < count; i += 1) {
      dots[i]!.setAttribute("fill", ramp((values[i]! - min) / span));
    }
  }

  // Conv sample blocks arrive as [4][k][k]; lay the 4 channels out 2×2.
  function paintConvSample(grid: DotGrid, block: Block, k: number): void {
    const range = trace.ranges[k === 5 ? "conv1_sample" : "conv2_sample"]!;
    const span = range.max - range.min || 1;
    for (let channel = 0; channel < 4; channel += 1) {
      const baseRow = Math.floor(channel / 2) * k;
      const baseCol = (channel % 2) * k;
      for (let r = 0; r < k; r += 1) {
        for (let c = 0; c < k; c += 1) {
          const value = block.values[channel * k * k + r * k + c]!;
          const dot = grid.dots[(baseRow + r) * grid.cols + (baseCol + c)]!;
          dot.setAttribute("fill", ramp((value - range.min) / span));
        }
      }
    }
  }

  function paintVector(grid: DotGrid, block: Block, rangeKey: string): void {
    const range = trace.ranges[rangeKey] ?? block;
    paint(grid.dots, block.values, range.min, range.max);
  }

  const missionActs = trace.mission_acts.mission_vec;
  paint(missionVec.dots, missionActs.values, missionActs.min, missionActs.max);

  let inspectorsAttached = false;
  function renderStep(t: number): void {
    const step = steps[t]!;
    // Observation planes, transposed so the agent sits on the bottom row.
    planes.forEach((grid, channel) => {
      for (let r = 0; r < 7; r += 1) {
        for (let c = 0; c < 7; c += 1) {
          const value = step.obs[c]![r]![channel]!;
          grid.dots[r * 7 + c]!.setAttribute(
            "fill", ramp(value / OBS_VMAX[channel]!),
          );
        }
      }
    });
    paintConvSample(conv1, step.acts["conv1_sample"]!, 5);
    paintConvSample(conv2, step.acts["conv2_sample"]!, 3);
    paintVector(obsVec, step.acts["obs_vec"]!, "obs_vec");
    paintVector(dirStrip, step.acts["dir_embed"]!, "dir_embed");
    paintVector(prevStrip, step.acts["prev_action_embed"]!, "prev_action_embed");
    paintVector(fusedVec, step.acts["fused"]!, "fused");
    paintVector(hiddenVec, step.acts["hidden"]!, "hidden");
    const previous = t > 0 ? steps[t - 1]!.acts["hidden"]! : null;
    const hiddenRange = trace.ranges["hidden"]!;
    if (previous) paint(ghostVec.dots, previous.values, hiddenRange.min, hiddenRange.max);
    else {
      const mid = ramp((0 - hiddenRange.min) / (hiddenRange.max - hiddenRange.min || 1));
      ghostVec.dots.forEach((dot) => dot.setAttribute("fill", mid));
    }
    prevChip.textContent =
      step.prev_action === null ? "<start>" : (names[step.prev_action] ?? "?");

    const logits = step.acts["logits"]!;
    const probs = step.acts["probs"]!.values;
    const logitRange = trace.ranges["logits"]!;
    const span = logitRange.max - logitRange.min || 1;
    logits.values.forEach((value, i) => {
      const dot = logitDots[i]!;
      dot.setAttribute("fill", ramp((value - logitRange.min) / span));
      dot.setAttribute("r", String(7 + probs[i]! * 6));
    });
    chosenRing.setAttribute("cy", String(108 + step.action * 30));
    logitLabels.forEach((el, i) => {
      el.setAttribute("font-weight", i === step.action ? "700" : "400");
    });
    if (step.oracle_label !== null && step.oracle_label !== step.action) {
      oracleMark.setAttribute("y", String(112 + step.oracle_label * 30));
      oracleMark.textContent = "← oracle's label";
    } else {
      oracleMark.textContent = "";
    }
    windowRect.setAttribute("opacity", "0");
    pulses.forEach((pulse) => pulse.setAttribute("opacity", "0"));

    const top = probs.indexOf(Math.max(...probs));
    readout.innerHTML =
      `Step ${step.t} of ${steps.length - 1}: the policy chose ` +
      `<b>${names[step.action]}</b> with probability ` +
      `${probs[top]!.toFixed(2)}` +
      (step.oracle_label !== null
        ? `, and the oracle labels this state <b>${names[step.oracle_label]}</b>.`
        : ".");

    if (!inspectorsAttached) {
      inspectorsAttached = true;
      const attach = (grid: DotGrid, name: string, rangeKey: string, count: number) => {
        const range = trace.ranges[rangeKey]!;
        inspectable(
          grid.dots[0]!.parentElement as unknown as SVGElement,
          name, range, count,
        );
      };
      attach(obsVec, "observation vector · 64", "obs_vec", 64);
      attach(fusedVec, "fused vector · 128", "fused", 128);
      attach(hiddenVec, "hidden state · 128", "hidden", 128);
      attach(conv1, "conv activations (4 sampled channels)", "conv1_sample", 100);
      legend.textContent =
        "fused vector · min " +
        `${trace.ranges["fused"]!.min} · max ${trace.ranges["fused"]!.max}`;
    }
  }

  // ---- one animated forward pass ------------------------------------------

  let stepTl: Timeline | null = null;
  function animateStep(t: number, onDone?: () => void): void {
    stepTl?.kill();
    renderStep(t);
    const step = steps[t]!;
    const tl = gsap.timeline({ onComplete: onDone });
    stepTl = tl;
    tl.fromTo(
      planes.map((p) => p.dots[0]!.parentElement),
      { opacity: 0.35 },
      { opacity: 1, duration: 0.3 },
      0,
    );
    // The 3×3 window glides over the object plane while conv dots light up.
    tl.set(windowRect, { opacity: 1 }, 0.15);
    const positions = [
      [0, 0], [2, 1], [4, 2], [2, 3], [0, 4],
    ].map(([c, r]) => ({ x: 22 + c! * 8 - 4, y: 46 + r! * 8 - 4 }));
    positions.forEach((pos, i) => {
      tl.to(windowRect, { attr: { x: pos.x, y: pos.y }, duration: 0.12 },
        0.15 + i * 0.12);
    });
    tl.to(windowRect, { opacity: 0, duration: 0.15 }, 0.85);
    tl.fromTo(
      [conv1.dots[0]!.parentElement, conv2.dots[0]!.parentElement],
      { opacity: 0.3 },
      { opacity: 1, duration: 0.3, stagger: 0.15 },
      0.35,
    );
    tl.fromTo(
      obsVec.dots, { opacity: 0.2 },
      { opacity: 1, duration: 0.25, stagger: { amount: 0.25 } }, 0.75,
    );
    tl.fromTo(
      [dirStrip.dots[0]!.parentElement, prevStrip.dots[0]!.parentElement],
      { opacity: 0.3 }, { opacity: 1, duration: 0.25 }, 0.85,
    );
    pulses.forEach((pulse, i) => {
      const path = [pathObs, pathMission, pathContext][i]!;
      tl.set(pulse, { opacity: 1 }, 1.05);
      tl.to(
        pulse,
        {
          duration: 0.35,
          ease: "power1.inOut",
          motionPath: { path },
        },
        1.05,
      );
      tl.set(pulse, { opacity: 0 }, 1.42);
    });
    tl.fromTo(
      fusedVec.dots, { opacity: 0.2 },
      { opacity: 1, duration: 0.25, stagger: { amount: 0.25 } }, 1.45,
    );
    tl.fromTo(
      ghostVec.dots[0]!.parentElement,
      { opacity: 0.25, x: -26 },
      { opacity: 1, x: 0, duration: 0.4, ease: "power1.out" },
      1.5,
    );
    tl.fromTo(
      hiddenVec.dots, { opacity: 0.2 },
      { opacity: 1, duration: 0.25, stagger: { amount: 0.25 } }, 1.8,
    );
    tl.fromTo(
      logitDots,
      { scale: 0.4, transformOrigin: "50% 50%" },
      { scale: 1, duration: 0.3, ease: "back.out(1.8)", stagger: 0.06 },
      2.05,
    );
    tl.fromTo(
      chosenRing,
      { scale: 0, transformOrigin: "50% 50%" },
      { scale: 1, duration: 0.3, ease: "back.out(2)" },
      2.25,
    );
  }

  // ---- controls (scrubber conventions) ------------------------------------

  const go = (t: number, animate: boolean) => {
    const clamped = Math.max(0, Math.min(steps.length - 1, t));
    slider.value = String(clamped);
    if (animate && !reduced) animateStep(clamped);
    else {
      stepTl?.kill();
      renderStep(clamped);
    }
  };

  let playing = false;
  let playTimer: number | null = null;
  const stopPlay = () => {
    playing = false;
    if (playTimer !== null) window.clearTimeout(playTimer);
    playTimer = null;
    playButton.textContent = "play";
  };
  const advance = () => {
    const next = Number(slider.value) + 1;
    if (!playing) return;
    if (next >= steps.length) {
      stopPlay();
      return;
    }
    slider.value = String(next);
    animateStep(next, () => {
      playTimer = window.setTimeout(advance, 500);
    });
  };

  slider.addEventListener("input", () => {
    stopPlay();
    go(Number(slider.value), false);
  });
  mount.querySelector('[data-ctl="back"]')!.addEventListener("click", () => {
    stopPlay();
    go(Number(slider.value) - 1, false);
  });
  mount.querySelector('[data-ctl="next"]')!.addEventListener("click", () => {
    stopPlay();
    go(Number(slider.value) + 1, !reduced);
  });
  if (reduced) {
    playButton.hidden = true;
  } else {
    playButton.addEventListener("click", () => {
      if (playing) {
        stopPlay();
        return;
      }
      playing = true;
      playButton.textContent = "pause";
      animateStep(Number(slider.value), () => {
        playTimer = window.setTimeout(advance, 500);
      });
    });
  }

  renderStep(0);

  return {
    setStep: (t: number, animate: boolean) => go(t, animate),
    destroy: () => {
      stopPlay();
      stepTl?.kill();
    },
  };
}
