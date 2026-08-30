// Every number on a page comes through this registry. Pages carry empty
// <span data-stat="…"> mounts; a key that is missing here, or data that fails
// validation upstream, leaves a visible gap instead of a silently wrong digit.

import { count, interval, percent, points, steps } from "../data/format";
import type { ExperimentSummary, JourneyData } from "../data/schema";

type Sources = { summary: ExperimentSummary; journey: JourneyData };
type Resolver = (sources: Sources) => string;

export const STATS: Record<string, Resolver> = {
  // study headline ---------------------------------------------------------
  "study.delta": ({ summary }) =>
    points(summary.primary_summary.mean_paired_difference),
  "study.interval": ({ summary }) =>
    interval(
      summary.primary_summary.interval.lower,
      summary.primary_summary.interval.upper,
    ),
  "study.bundles": ({ summary }) =>
    String(summary.primary_summary.pipeline_replicates),
  "study.eligible": ({ summary }) =>
    count(summary.eligibility.eligible_scenarios),
  "study.candidates": ({ summary }) =>
    count(summary.eligibility.candidate_scenarios),
  "study.budget": ({ summary }) =>
    count(summary.budget.additional_revealed_targets.recovery),
  "study.sesoi": ({ summary }) =>
    points(summary.protocol.sesoi_absolute_success).replace("+", ""),
  "study.sensitivity": ({ summary }) =>
    interval(
      summary.primary_summary.sensitivity.lower,
      summary.primary_summary.sensitivity.upper,
    ),
  "study.precision": ({ summary }) =>
    points(summary.primary_summary.precision.achieved_half_width).replace("+", "±"),
  "study.protocol": ({ summary }) => `protocol ${summary.protocol.version}`,

  // environment ------------------------------------------------------------
  "env.id": ({ journey }) => journey.environment.env_id,
  "env.max-steps": ({ journey }) => String(journey.environment.max_steps),
  "env.grid": ({ journey }) => journey.lab01.grid_shape,
  "env.actions": ({ journey }) => journey.environment.action_names.join(", "),

  // lab 01 -----------------------------------------------------------------
  "lab01.census": ({ journey }) => count(journey.lab01.census_seeds),
  "lab01.missions": ({ journey }) => String(journey.lab01.unique_missions),

  // lab 02 -----------------------------------------------------------------
  "lab02.states": ({ journey }) => count(journey.lab02.total_states),
  "lab02.episodes": ({ journey }) => count(journey.lab02.episodes),
  "lab02.classes": ({ journey }) => count(journey.lab02.observation_classes),
  "lab02.aliased": ({ journey }) => String(journey.lab02.aliased_classes),
  "lab02.conflicting": ({ journey }) => String(journey.lab02.conflicting_classes),
  "lab02.floor": ({ journey }) => percent(journey.lab02.memoryless_error_floor),

  // lab 03 -----------------------------------------------------------------
  "lab03.random": ({ journey }) =>
    percent(journey.lab03.spectrum.random?.success_rate ?? NaN),
  "lab03.wall": ({ journey }) =>
    percent(journey.lab03.spectrum.wall_follower?.success_rate ?? NaN),
  "lab03.oracle": ({ journey }) =>
    percent(journey.lab03.spectrum.oracle?.success_rate ?? NaN, 0),
  "lab03.oracle-steps": ({ journey }) =>
    steps(journey.lab03.spectrum.oracle?.mean_steps ?? NaN),
  "lab03.random-steps": ({ journey }) =>
    steps(journey.lab03.spectrum.random?.mean_steps ?? NaN),
  "lab03.sync-pairs": ({ journey }) => String(journey.lab03.sync.pairs),
  "lab03.recovery-rate": ({ journey }) =>
    percent(journey.lab03.sync.success_rates.honest ?? NaN, 0),

  // lab 04 -----------------------------------------------------------------
  "lab04.q-states": ({ journey }) => String(journey.lab04.qlearning.states),
  "lab04.q-steps": ({ journey }) => String(journey.lab04.qlearning.best_steps),
  "lab04.demos": ({ journey }) => String(journey.lab04.dataset.episodes),
  "lab04.labels": ({ journey }) => count(journey.lab04.dataset.labels),
  "lab04.mem-acc": ({ journey }) =>
    percent(journey.lab04.memoryless.open_loop_accuracy_mean),
  "lab04.mem-unseen": ({ journey }) =>
    percent(journey.lab04.memoryless.unseen_success_mean),
  "lab04.rec-acc": ({ journey }) =>
    percent(journey.lab04.recurrent.open_loop_accuracy_mean),
  "lab04.rec-unseen": ({ journey }) =>
    percent(journey.lab04.recurrent.unseen_success_mean),

  // lab 05 -----------------------------------------------------------------
  "lab05.parameters": ({ journey }) => count(journey.lab05.parameters),

  // lab 06 -----------------------------------------------------------------
  "lab06.budget": ({ journey }) => count(journey.lab06.design.budget_labels),
  "lab06.deltas": ({ journey }) =>
    journey.lab06.per_rep_unseen_delta.map((d) => points(d)).join(", "),
  "lab06.delivered": ({ journey }) => percent(journey.lab06.delivered_rate),
  "lab06.exposure-extra": ({ journey }) =>
    (journey.lab06.exposure.extra ?? NaN).toFixed(1),
  "lab06.exposure-recovery": ({ journey }) =>
    (journey.lab06.exposure.recovery ?? NaN).toFixed(1),
  "lab06.base-clean": ({ journey }) =>
    percent(journey.lab06.success_matrix.base?.clean ?? NaN),
  "lab06.base-matched": ({ journey }) =>
    percent(journey.lab06.success_matrix.base?.matched ?? NaN),

  // lab 07 -----------------------------------------------------------------
  "lab07.truth": ({ journey }) =>
    points(journey.lab07.simulation.true_itt_effect ?? NaN),
  "lab07.itt-bias": ({ journey }) =>
    points(journey.lab07.simulation.itt_bias ?? NaN, 2),
  "lab07.pp-bias": ({ journey }) =>
    points(journey.lab07.simulation.per_protocol_bias ?? NaN, 2),
  "lab07.paired-power": ({ journey }) =>
    percent(journey.lab07.pairing.paired_power ?? NaN, 0),
  "lab07.unpaired-power": ({ journey }) =>
    percent(journey.lab07.pairing.unpaired_power ?? NaN, 0),
  "lab07.unmatched-budget": ({ journey }) =>
    journey.lab07.unmatched ? count(journey.lab07.unmatched.budget_labels) : "n/a",
  "lab07.unmatched-unseen": ({ journey }) =>
    journey.lab07.unmatched
      ? percent(journey.lab07.unmatched.unseen_success)
      : "n/a",
  "lab07.matched-recovery-unseen": ({ journey }) =>
    percent(journey.lab06.success_matrix.recovery?.unseen ?? NaN),
};

export function bindStats(sources: Sources, root: ParentNode = document): void {
  for (const span of root.querySelectorAll<HTMLElement>("[data-stat]")) {
    const key = span.dataset.stat ?? "";
    const resolver = STATS[key];
    if (!resolver) {
      span.textContent = "n/a";
      span.classList.add("stat-missing");
      console.error(`unknown stat key: ${key}`);
      continue;
    }
    span.textContent = resolver(sources);
  }
}
