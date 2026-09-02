// Accounting invariants, enforced before any number reaches the page. A
// malformed bundle produces a visible "results unavailable" state, never a
// partially rendered result.

import type { ExperimentSummary, SiteStatus, SliceId } from "./schema";

const PRIMARY_SLICE: SliceId = "unseen";
const RATE_TOLERANCE = 1e-9;
const MEAN_TOLERANCE = 1e-9;

export function validateStatus(status: SiteStatus): string[] {
  const problems: string[] = [];
  const phases = ["protocol", "pilot", "results"];
  if (!phases.includes(status.phase)) {
    problems.push(`unknown phase "${status.phase}"`);
  }
  if (status.result_release && status.phase !== "results") {
    problems.push("result_release without results phase");
  }
  if (status.phase === "results" && !status.protocol_hash) {
    problems.push("results phase without a protocol hash");
  }
  return problems;
}

export function validateSummary(
  summary: ExperimentSummary,
  status: SiteStatus,
): string[] {
  const problems: string[] = [];
  const finite = (value: number, name: string) => {
    if (!Number.isFinite(value)) problems.push(`${name} is not finite`);
  };

  if (status.protocol_hash && summary.protocol.hash !== status.protocol_hash) {
    problems.push("summary protocol hash differs from the released status hash");
  }

  const budget = summary.budget.additional_revealed_targets;
  if (budget.extra_demo !== budget.recovery) {
    problems.push("label budgets differ between arms");
  }
  if (
    !summary.budget.optimizer_updates_matched ||
    !summary.budget.target_exposures_matched ||
    !summary.budget.replay_rules_matched
  ) {
    problems.push("matching flags are not all true");
  }

  if (
    summary.primary_summary.analysis_status === "confirmatory" &&
    summary.replicates.length < 5
  ) {
    problems.push("confirmatory status with fewer than five bundles");
  }
  if (summary.primary_summary.pipeline_replicates !== summary.replicates.length) {
    problems.push("replicate count disagrees with the replicate list");
  }

  const deltas: number[] = [];
  for (const replicate of summary.replicates) {
    const denominators = new Set<number>();
    for (const outcome of replicate.outcomes) {
      finite(outcome.success_rate, `${replicate.bundle_id} success rate`);
      const recomputed = outcome.successful_episodes / outcome.assigned_episodes;
      if (Math.abs(recomputed - outcome.success_rate) > RATE_TOLERANCE) {
        problems.push(
          `${replicate.bundle_id}/${outcome.method}/${outcome.slice}: ` +
            "success rate is not successes over assigned episodes",
        );
      }
      if (
        outcome.intervention_delivered !== null &&
        outcome.intervention_delivered > outcome.assigned_episodes
      ) {
        problems.push(
          `${replicate.bundle_id}/${outcome.method}/${outcome.slice}: ` +
            "delivered exceeds assigned (ITT denominator broken)",
        );
      }
      denominators.add(outcome.assigned_episodes);
    }
    if (denominators.size !== 1) {
      problems.push(
        `${replicate.bundle_id}: methods face different scenario denominators`,
      );
    }
    const bySlice = (method: string) =>
      replicate.outcomes.find(
        (o) => o.method === method && o.slice === PRIMARY_SLICE,
      );
    const recovery = bySlice("recovery");
    const extra = bySlice("extra_demo");
    if (recovery && extra) {
      const delta = recovery.success_rate - extra.success_rate;
      deltas.push(delta);
      if (Math.abs(delta - replicate.primary_paired_difference) > RATE_TOLERANCE) {
        problems.push(
          `${replicate.bundle_id}: stored paired difference is not the ` +
            "unseen recovery-minus-extra difference",
        );
      }
    } else {
      problems.push(`${replicate.bundle_id}: missing primary-slice outcomes`);
    }
  }

  if (deltas.length > 0) {
    const mean = deltas.reduce((a, b) => a + b, 0) / deltas.length;
    if (
      Math.abs(mean - summary.primary_summary.mean_paired_difference) >
      MEAN_TOLERANCE
    ) {
      problems.push(
        "displayed mean is not recomputable from the replicate differences",
      );
    }
  }

  const interval = summary.primary_summary.interval;
  finite(interval.lower, "interval lower");
  finite(interval.upper, "interval upper");
  if (interval.lower > interval.upper) {
    problems.push("interval bounds are inverted");
  }
  finite(summary.protocol.sesoi_absolute_success, "SESOI");

  return problems;
}

// The preregistered wording, selected mechanically from the claim state.
export function claimHeadline(summary: ExperimentSummary): string {
  switch (summary.primary_summary.claim_state) {
    case "support":
      return "The prespecified interval excludes zero in favor of recovery labels.";
    case "adverse":
      return "The prespecified interval excludes zero in favor of extra demonstrations.";
    case "rule_out":
      return "The prespecified interval rules out effects as large as the SESOI in either direction.";
    default:
      return "The prespecified interval does not resolve the comparison.";
  }
}
