// @vitest-environment node
// The validator must accept the real released bundle and reject every
// tampered variant it is designed to catch.

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { ExperimentSummary, SiteStatus } from "../src/data/schema";
import {
  claimHeadline,
  validateStatus,
  validateSummary,
} from "../src/data/validate";

const summarySource = JSON.parse(
  readFileSync(new URL("../../public_result/experiment-summary.json", import.meta.url), "utf-8"),
) as ExperimentSummary;
const statusSource = JSON.parse(
  readFileSync(new URL("../../public_result/site-status.json", import.meta.url), "utf-8"),
) as SiteStatus;

function cloneSummary(): ExperimentSummary {
  return structuredClone(summarySource);
}

describe("released bundle", () => {
  it("passes status validation", () => {
    expect(validateStatus(statusSource)).toEqual([]);
  });

  it("passes summary validation", () => {
    expect(validateSummary(summarySource, statusSource)).toEqual([]);
  });

  it("selects the support headline mechanically", () => {
    expect(claimHeadline(summarySource)).toMatch(/recovery labels/);
  });
});

describe("tampered bundles are refused", () => {
  it("catches a hand-edited mean", () => {
    const summary = cloneSummary();
    summary.primary_summary.mean_paired_difference += 0.01;
    expect(validateSummary(summary, statusSource)).toContainEqual(
      expect.stringContaining("recomputable"),
    );
  });

  it("catches a success rate that is not successes over assigned", () => {
    const summary = cloneSummary();
    summary.replicates[0]!.outcomes[0]!.success_rate += 0.001;
    expect(
      validateSummary(summary, statusSource).some((p) =>
        p.includes("not successes over assigned"),
      ),
    ).toBe(true);
  });

  it("catches unequal label budgets", () => {
    const summary = cloneSummary();
    summary.budget.additional_revealed_targets.recovery *= 2;
    expect(validateSummary(summary, statusSource)).toContainEqual(
      expect.stringContaining("budgets differ"),
    );
  });

  it("catches unmatched optimization flags", () => {
    const summary = cloneSummary();
    summary.budget.optimizer_updates_matched = false;
    expect(validateSummary(summary, statusSource)).toContainEqual(
      expect.stringContaining("matching flags"),
    );
  });

  it("catches a paired difference that is not the unseen contrast", () => {
    const summary = cloneSummary();
    summary.replicates[0]!.primary_paired_difference += 0.02;
    expect(
      validateSummary(summary, statusSource).some((p) =>
        p.includes("paired difference"),
      ),
    ).toBe(true);
  });

  it("catches confirmatory status with too few bundles", () => {
    const summary = cloneSummary();
    summary.replicates = summary.replicates.slice(0, 4);
    summary.primary_summary.pipeline_replicates = 4;
    const problems = validateSummary(summary, statusSource);
    expect(problems.some((p) => p.includes("fewer than five"))).toBe(true);
  });

  it("catches a protocol hash mismatch", () => {
    const summary = cloneSummary();
    summary.protocol.hash = "0".repeat(64);
    expect(validateSummary(summary, statusSource)).toContainEqual(
      expect.stringContaining("protocol hash"),
    );
  });

  it("catches a release flag without results phase", () => {
    const status = structuredClone(statusSource);
    status.phase = "pilot";
    expect(validateStatus(status)).toContainEqual(
      expect.stringContaining("result_release"),
    );
  });
});
