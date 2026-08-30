// The three-arm allocation diagram: one shared checkpoint branching into the
// arms, each described directly underneath its own box, with the quantities
// held equal across all three stated once below the whole graphic.

import { count } from "../data/format";
import type { ExperimentSummary } from "../data/schema";

const ARMS = [
  {
    id: "bc_base",
    cls: "node--base",
    label: "BC base",
    detail: "no additional labels, the contextual baseline",
    varies: "receives nothing",
  },
  {
    id: "extra_demo",
    cls: "node--extra",
    label: "Extra demonstrations",
    detail: "budget spent on fresh expert episodes",
    varies: "where the labels come from: the expert's own states",
  },
  {
    id: "recovery",
    cls: "node--recovery",
    label: "Recovery aggregation",
    detail: "budget spent on learner-visited post-corruption states",
    varies: "where the labels come from: states the learner actually reaches",
  },
] as const;

export function renderThreeArm(
  mount: HTMLElement,
  summary: ExperimentSummary,
): void {
  const budget = count(summary.budget.additional_revealed_targets.recovery);
  const wrapper = document.createElement("div");
  wrapper.className = "arm-diagram";
  wrapper.innerHTML = `
    <svg viewBox="0 0 960 270" role="img">
      <title>Three arms from one checkpoint</title>
      <desc>One trained base checkpoint is cloned into three arms; the two
        augmented arms receive equal label budgets that differ only in where
        the labels are collected.</desc>
      <rect class="node" x="378" y="16" width="204" height="56" rx="8"></rect>
      <text x="480" y="40" text-anchor="middle">one base checkpoint</text>
      <text x="480" y="60" text-anchor="middle" class="detail">cloned bit-exactly, three times</text>
      ${ARMS.map((arm, index) => {
        const x = 60 + index * 300;
        return `
          <path class="edge" d="M 480 72 C 480 108, ${x + 120} 118, ${x + 120} 150"></path>
          <g class="arm">
            <rect class="node ${arm.cls}" x="${x}" y="150" width="240"
                  height="70" rx="8" stroke-width="2"></rect>
            <text x="${x + 120}" y="178" text-anchor="middle">${arm.label}</text>
            <text x="${x + 120}" y="200" text-anchor="middle" class="detail">
              ${arm.id === "bc_base" ? "+0 labels" : `+${budget} labels`}</text>
          </g>`;
      }).join("")}
    </svg>
    <div class="arm-legend">
      ${ARMS.map(
        (arm) => `
        <p data-arm="${arm.id}">
          <b>${arm.label}.</b> ${arm.detail}.
          <span class="muted">Varies: ${arm.varies}.</span>
        </p>`,
      ).join("")}
    </div>
    <p class="arm-shared small">
      <b>Held equal everywhere:</b> starting weights, optimizer and update
      count, per-update target exposure, replay rules, evaluation scenarios.
    </p>
  `;

  mount.replaceChildren(wrapper);
}
