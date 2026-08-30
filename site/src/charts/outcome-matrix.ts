// Success by evaluation slice and method, averaged over bundles, with the
// exact denominators stated and the per-bundle spread shown as dots.

import { percent } from "../data/format";
import type { ExperimentSummary, MethodId, SliceId } from "../data/schema";

const SVG_NS = "http://www.w3.org/2000/svg";
const METHODS: Array<{ id: MethodId; label: string; color: string }> = [
  { id: "bc_base", label: "base", color: "var(--base-method)" },
  { id: "extra_demo", label: "extra demos", color: "var(--extra)" },
  { id: "recovery", label: "recovery", color: "var(--recovery)" },
];
const SLICES: Array<{ id: SliceId; label: string }> = [
  { id: "clean", label: "clean" },
  { id: "matched", label: "matched corruption" },
  { id: "unseen", label: "unseen corruption" },
];

export function renderOutcomeMatrix(
  mount: HTMLElement,
  summary: ExperimentSummary,
): void {
  const rates = new Map<string, number[]>();
  let denominator = 0;
  for (const replicate of summary.replicates) {
    for (const outcome of replicate.outcomes) {
      const key = `${outcome.method}/${outcome.slice}`;
      rates.set(key, [...(rates.get(key) ?? []), outcome.success_rate]);
      denominator = outcome.assigned_episodes;
    }
  }
  const mean = (values: number[]) =>
    values.reduce((a, b) => a + b, 0) / values.length;

  const width = 960;
  const groupWidth = 288;
  const barWidth = 64;
  const chartTop = 16;
  const chartBottom = 210;
  const height = 264;
  const y = (rate: number) =>
    chartBottom - rate * (chartBottom - chartTop);

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  const title = document.createElementNS(SVG_NS, "title");
  title.textContent = "Success by slice and method";
  svg.append(title);

  SLICES.forEach((slice, sliceIndex) => {
    const groupX = 36 + sliceIndex * (groupWidth + 24);
    METHODS.forEach((method, methodIndex) => {
      const values = rates.get(`${method.id}/${slice.id}`) ?? [];
      if (values.length === 0) return;
      const barX = groupX + methodIndex * (barWidth + 24);
      const meanRate = mean(values);
      const bar = document.createElementNS(SVG_NS, "rect");
      bar.setAttribute("x", String(barX));
      bar.setAttribute("y", String(y(meanRate)));
      bar.setAttribute("width", String(barWidth));
      bar.setAttribute("height", String(chartBottom - y(meanRate)));
      bar.setAttribute("fill", method.color);
      bar.setAttribute("rx", "3");
      svg.append(bar);
      for (const value of values) {
        const dot = document.createElementNS(SVG_NS, "circle");
        dot.setAttribute("cx", String(barX + barWidth / 2));
        dot.setAttribute("cy", String(y(value)));
        dot.setAttribute("r", "3");
        dot.setAttribute("fill", "var(--ink)");
        svg.append(dot);
      }
      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute("x", String(barX + barWidth / 2));
      // clear of the highest per-bundle dot, so the two never overprint
      const highest = Math.min(y(Math.max(...values)) - 8, y(meanRate) - 6);
      label.setAttribute("y", String(highest));
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("font-size", "13");
      label.textContent = percent(meanRate, 0);
      svg.append(label);
    });
    const groupLabel = document.createElementNS(SVG_NS, "text");
    groupLabel.setAttribute("x", String(groupX + groupWidth / 2 - 12));
    groupLabel.setAttribute("y", String(chartBottom + 22));
    groupLabel.setAttribute("text-anchor", "middle");
    groupLabel.setAttribute("font-size", "14");
    groupLabel.textContent = slice.label;
    svg.append(groupLabel);
  });

  const chart = document.createElement("div");
  chart.className = "chart";
  chart.append(svg);

  const legend = document.createElement("div");
  legend.className = "legend";
  legend.innerHTML = METHODS.map(
    (m) =>
      `<span><span class="swatch" style="background:${m.color}"></span>${m.label}</span>`,
  ).join("");
  const note = document.createElement("p");
  note.className = "small muted";
  note.textContent =
    `Bars: mean over ${summary.replicates.length} bundles; dots: per-bundle ` +
    `values; every cell evaluates the same ${denominator} scenarios, ` +
    "intention-to-treat.";

  const table = document.createElement("div");
  table.className = "table-scroll";
  table.innerHTML = `
    <table>
      <caption class="small muted">Mean success by slice and method.</caption>
      <thead><tr><th scope="col">method</th>${SLICES.map(
        (s) => `<th scope="col">${s.label}</th>`,
      ).join("")}</tr></thead>
      <tbody>${METHODS.map(
        (m) =>
          `<tr><td>${m.label}</td>${SLICES.map((s) => {
            const values = rates.get(`${m.id}/${s.id}`) ?? [];
            return `<td>${values.length ? percent(mean(values)) : "n/a"}</td>`;
          }).join("")}</tr>`,
      ).join("")}</tbody>
    </table>
  `;
  mount.replaceChildren(chart, legend, note, table);
}
