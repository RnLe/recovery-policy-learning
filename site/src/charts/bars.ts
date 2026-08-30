// A small horizontal-bar builder used by the journey pages (policy spectrum,
// ablation, mini three-arm). Direct labels, visible values, table fallback.

import { percent } from "../data/format";

const SVG_NS = "http://www.w3.org/2000/svg";

export type BarRow = {
  label: string;
  value: number; // 0..1
  color: string; // CSS custom property or color
  detail?: string;
  dots?: number[]; // e.g. per-seed values
};

export function renderBars(
  mount: HTMLElement,
  rows: BarRow[],
  options: { title: string; description: string },
): void {
  const width = 960;
  const rowHeight = 40;
  const left = 220;
  const right = width - 84;
  const height = rows.length * rowHeight + 12;
  const x = (value: number) => left + value * (right - left);

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  const title = document.createElementNS(SVG_NS, "title");
  title.textContent = options.title;
  const description = document.createElementNS(SVG_NS, "desc");
  description.textContent = options.description;
  svg.append(title, description);

  rows.forEach((row, index) => {
    const y = index * rowHeight + 8;
    const bar = document.createElementNS(SVG_NS, "rect");
    bar.setAttribute("x", String(left));
    bar.setAttribute("y", String(y));
    bar.setAttribute("width", String(Math.max(1, x(row.value) - left)));
    bar.setAttribute("height", "22");
    bar.setAttribute("rx", "3");
    bar.setAttribute("fill", row.color);
    svg.append(bar);

    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("x", String(left - 10));
    label.setAttribute("y", String(y + 16));
    label.setAttribute("text-anchor", "end");
    label.setAttribute("font-size", "14");
    label.setAttribute("fill", "var(--ink)");
    label.textContent = row.label;
    svg.append(label);

    for (const dot of row.dots ?? []) {
      const mark = document.createElementNS(SVG_NS, "circle");
      mark.setAttribute("cx", String(x(dot)));
      mark.setAttribute("cy", String(y + 11));
      mark.setAttribute("r", "4");
      mark.setAttribute("fill", "var(--ink)");
      svg.append(mark);
    }

    const value = document.createElementNS(SVG_NS, "text");
    value.setAttribute("x", String(right + 12));
    value.setAttribute("y", String(y + 16));
    value.setAttribute("font-size", "14");
    value.setAttribute("fill", "var(--ink-soft)");
    value.textContent = percent(row.value);
    svg.append(value);
  });

  const chart = document.createElement("div");
  chart.className = "chart";
  chart.append(svg);

  const table = document.createElement("div");
  table.className = "table-scroll";
  table.innerHTML = `
    <table>
      <caption class="small muted">${options.title}</caption>
      <thead><tr><th scope="col">condition</th><th scope="col">success</th>
        <th scope="col">note</th></tr></thead>
      <tbody>${rows
        .map(
          (row) =>
            `<tr><td>${row.label}</td><td>${percent(row.value)}</td>` +
            `<td>${row.detail ?? ""}</td></tr>`,
        )
        .join("")}</tbody>
    </table>
  `;
  mount.replaceChildren(chart, table);
}
