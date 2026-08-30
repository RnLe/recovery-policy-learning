// The primary chart: one dot per pipeline bundle, the mean with its 95%
// interval, and the zero/SESOI references. Every value also lands in an
// adjacent table so nothing lives only in the graphic.

import { points } from "../data/format";
import type { ExperimentSummary } from "../data/schema";

const SVG_NS = "http://www.w3.org/2000/svg";

function element<K extends keyof SVGElementTagNameMap>(
  name: K,
  attributes: Record<string, string | number>,
  text?: string,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    node.setAttribute(key, String(value));
  }
  if (text !== undefined) node.textContent = text;
  return node;
}

export function renderPairedEffect(
  mount: HTMLElement,
  summary: ExperimentSummary,
): void {
  const primary = summary.primary_summary;
  const deltas = summary.replicates.map((r) => ({
    bundle: r.bundle_id,
    delta: r.primary_paired_difference,
  }));
  const sesoi = summary.protocol.sesoi_absolute_success;

  const width = 960;
  const height = 96 + deltas.length * 30;
  const left = 90;
  const right = width - 20;
  const values = [
    0,
    sesoi,
    -sesoi,
    primary.interval.lower,
    primary.interval.upper,
    ...deltas.map((d) => d.delta),
  ];
  const min = Math.min(...values) - 0.02;
  const max = Math.max(...values) + 0.02;
  const x = (value: number) => left + ((value - min) / (max - min)) * (right - left);

  const svg = element("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-labelledby": "paired-effect-title paired-effect-desc",
  });
  svg.append(
    element("title", { id: "paired-effect-title" }, "Paired primary effect"),
    element(
      "desc",
      { id: "paired-effect-desc" },
      "Per-bundle recovery-minus-extra differences with the mean and its 95% interval; the table below lists the same values.",
    ),
  );

  const axisY = height - 34;
  svg.append(
    element("line", {
      x1: left, x2: right, y1: axisY, y2: axisY,
      stroke: "var(--line-strong)",
    }),
  );
  for (const tick of [-0.05, 0, 0.05, 0.1, 0.15, 0.2]) {
    if (tick < min || tick > max) continue;
    svg.append(
      element("line", {
        x1: x(tick), x2: x(tick), y1: axisY, y2: axisY + 5,
        stroke: "var(--line-strong)",
      }),
      element(
        "text",
        { x: x(tick), y: axisY + 20, "text-anchor": "middle", "font-size": 13 },
        points(tick, 0),
      ),
    );
  }

  svg.append(
    element("line", {
      x1: x(0), x2: x(0), y1: 12, y2: axisY, stroke: "var(--ink-faint)",
    }),
  );
  for (const guard of [sesoi, -sesoi]) {
    if (guard < min || guard > max) continue;
    svg.append(
      element("line", {
        x1: x(guard), x2: x(guard), y1: 12, y2: axisY,
        stroke: "var(--caution)", "stroke-dasharray": "4 4",
      }),
    );
  }
  svg.append(
    element(
      "text",
      { x: x(sesoi) + 5, y: 22, "font-size": 12.5, fill: "var(--caution-strong)" },
      "SESOI",
    ),
  );

  deltas.forEach((entry, index) => {
    const y = 36 + index * 30;
    svg.append(
      element(
        "text",
        { x: left - 12, y: y + 5, "text-anchor": "end", "font-size": 13 },
        entry.bundle,
      ),
      element("circle", {
        cx: x(entry.delta), cy: y, r: 6, fill: "var(--recovery)",
      }),
    );
  });

  const meanY = 36 + deltas.length * 30 - 8;
  svg.append(
    element("line", {
      x1: x(primary.interval.lower), x2: x(primary.interval.upper),
      y1: meanY, y2: meanY, stroke: "var(--ink)", "stroke-width": 2,
    }),
    element("rect", {
      x: x(primary.mean_paired_difference) - 6, y: meanY - 6,
      width: 12, height: 12, fill: "var(--ink)",
      transform: `rotate(45 ${x(primary.mean_paired_difference)} ${meanY})`,
    }),
    element(
      "text",
      { x: left - 12, y: meanY + 5, "text-anchor": "end", "font-size": 13,
        "font-weight": 700 },
      "mean",
    ),
  );

  const chart = document.createElement("div");
  chart.className = "chart";
  chart.append(svg);

  const table = document.createElement("div");
  table.className = "table-scroll";
  table.innerHTML = `
    <table>
      <caption class="small muted">The same values as the chart.</caption>
      <thead><tr><th scope="col">bundle</th>
        <th scope="col">recovery − extra (unseen, ITT)</th></tr></thead>
      <tbody>
        ${deltas
          .map(
            (d) => `<tr><td>${d.bundle}</td><td>${points(d.delta, 2)}</td></tr>`,
          )
          .join("")}
        <tr><th scope="row">mean (95% CI)</th>
          <td>${points(primary.mean_paired_difference, 2)}
            &nbsp;[${points(primary.interval.lower, 2)},
            ${points(primary.interval.upper, 2)}]</td></tr>
      </tbody>
    </table>
  `;
  mount.replaceChildren(chart, table);
}
