// Components rendered against the real released bundle: the right numbers
// must appear, the gates must hold, and the interactive states must move.

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { renderBars } from "../src/charts/bars";
import { renderOutcomeMatrix } from "../src/charts/outcome-matrix";
import { renderPairedEffect } from "../src/charts/paired-effect";
import { renderResultPanel } from "../src/components/result-panel";
import { renderSliceExplainer } from "../src/components/slice-explainer";
import { renderThreeArm } from "../src/components/three-arm";
import type { ExperimentSummary, SiteStatus } from "../src/data/schema";

const bundle = (name: string) =>
  resolve(process.cwd(), "..", "public_result", name);
const summary = JSON.parse(
  readFileSync(bundle("experiment-summary.json"), "utf-8"),
) as ExperimentSummary;
const status = JSON.parse(
  readFileSync(bundle("site-status.json"), "utf-8"),
) as SiteStatus;

describe("result panel", () => {
  it("renders the released numbers", () => {
    const mount = document.createElement("div");
    renderResultPanel(mount, status, summary);
    expect(mount.textContent).toContain("+14.3 pp");
    expect(mount.textContent).toContain("[+10.0 pp, +18.6 pp]");
    expect(mount.textContent).toContain("536");
    expect(mount.textContent).toContain("recovery labels");
  });

  it("refuses to render numbers before release", () => {
    const mount = document.createElement("div");
    renderResultPanel(
      mount,
      { ...status, phase: "protocol", result_release: false },
      summary,
    );
    expect(mount.textContent).toContain("Protocol only");
    expect(mount.textContent).not.toContain("pp");
  });
});

describe("paired-effect chart", () => {
  it("draws one dot per bundle plus the mean, and mirrors them in a table", () => {
    const mount = document.createElement("div");
    renderPairedEffect(mount, summary);
    expect(mount.querySelectorAll("circle").length).toBe(
      summary.replicates.length,
    );
    const rows = mount.querySelectorAll("tbody tr");
    expect(rows.length).toBe(summary.replicates.length + 1);
    expect(mount.textContent).toContain("SESOI");
  });
});

describe("outcome matrix", () => {
  it("draws nine bars with denominators stated", () => {
    const mount = document.createElement("div");
    renderOutcomeMatrix(mount, summary);
    expect(mount.querySelectorAll("rect").length).toBe(9);
    expect(mount.textContent).toContain("536 scenarios");
  });
});

describe("three-arm diagram", () => {
  it("describes every arm under its box, with one shared matched line", () => {
    const mount = document.createElement("div");
    renderThreeArm(mount, summary);
    expect(mount.querySelectorAll(".arm-legend [data-arm]").length).toBe(3);
    expect(mount.querySelector('[data-arm="recovery"]')!.textContent).toContain(
      "states the learner actually reaches",
    );
    expect(mount.querySelector(".arm-shared")!.textContent).toContain(
      "Held equal",
    );
    expect(mount.textContent).toContain("1,000");
  });
});

describe("slice explainer", () => {
  it("switches panels and keeps exactly one visible", () => {
    const mount = document.createElement("div");
    renderSliceExplainer(mount);
    const tabs = mount.querySelectorAll<HTMLButtonElement>("[role=tab]");
    expect(tabs.length).toBe(3);
    tabs[2]!.click();
    const panels = [...mount.querySelectorAll<HTMLElement>("[role=tabpanel]")];
    expect(panels.filter((p) => !p.hidden).length).toBe(1);
    expect(panels[2]!.hidden).toBe(false);
    expect(panels[2]!.textContent).toContain("cannot support");
  });
});

describe("bars", () => {
  it("renders values, dots, and the table fallback", () => {
    const mount = document.createElement("div");
    renderBars(
      mount,
      [
        { label: "a", value: 0.5, color: "red", dots: [0.4, 0.6], detail: "x" },
        { label: "b", value: 0.8, color: "blue" },
      ],
      { title: "t", description: "d" },
    );
    expect(mount.querySelectorAll("rect").length).toBe(2);
    expect(mount.querySelectorAll("circle").length).toBe(2);
    expect(mount.textContent).toContain("80.0%");
    expect(mount.querySelectorAll("tbody tr").length).toBe(2);
  });
});

describe("story player (static path)", () => {
  it("shows every section at once and steps on click", async () => {
    const { mountStoryPlayer } = await import("../src/components/story-player");
    const { driftStory } = await import("../src/components/stories/drift-story");
    const mount = document.createElement("div");
    document.body.append(mount);
    mountStoryPlayer(mount, driftStory, { captionVariant: "hero" });
    const steps = mount.querySelectorAll<HTMLButtonElement>(".story__step");
    expect(steps.length).toBe(4);
    expect(steps[0]!.getAttribute("aria-current")).toBe("step");
    // Every explanation is on the page at once; the story loops on its own, so
    // there is no replay control and nothing is announced as it changes.
    expect(mount.textContent).toContain("The expert's path");
    expect(mount.textContent).toContain("Two ways to spend labels");
    expect(mount.querySelector("[aria-live]")).toBeNull();
    expect(mount.querySelector(".story__toggle")).toBeNull();
    // jsdom has no matchMedia, so the player takes the reduced-motion path:
    // sections step statically through the buttons.
    steps[3]!.click();
    expect(steps[3]!.getAttribute("aria-current")).toBe("step");
    expect(steps[0]!.hasAttribute("aria-current")).toBe(false);
    const fills = mount.querySelectorAll<HTMLElement>(".story__bar-fill");
    expect([...fills].every((fill) => fill.style.transform === "scaleX(1)")).toBe(
      true,
    );
    const groups = mount.querySelectorAll<SVGGElement>("svg > g");
    expect([...groups].every((g) => g.style.opacity === "1")).toBe(true);
    mount.remove();
  });
});

describe("network flow (static path)", () => {
  it("renders real values through the ramp with a working slider", async () => {
    const { createNetworkFlow } = await import("../src/components/network-flow");
    const trace = JSON.parse(
      readFileSync(
        resolve(
          process.cwd(), "..", "foundations", "media", "network", "full_r0.json",
        ),
        "utf-8",
      ),
    );
    const mount = document.createElement("div");
    document.body.append(mount);
    createNetworkFlow(mount, trace);
    // one dot per exported value in the fixed blocks
    const circles = mount.querySelectorAll("circle");
    expect(circles.length).toBeGreaterThan(1000);
    expect(mount.textContent).toContain("the policy chose");
    // scrub to the last step; the readout follows
    const slider = mount.querySelector<HTMLInputElement>("input[type=range]")!;
    expect(Number(slider.max)).toBe(trace.steps.length - 1);
    slider.value = slider.max;
    slider.dispatchEvent(new Event("input"));
    expect(mount.textContent).toContain(`Step ${trace.steps.length - 1} of`);
    // the legend shows the range the trace actually holds
    expect(mount.textContent).toContain(String(trace.ranges.fused.max));
    mount.remove();
  });
});
