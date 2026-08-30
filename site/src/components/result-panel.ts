// The confirmatory result, rendered only from the validated summary and only
// when the release gate is open. Same layout regardless of the result's sign.

import { interval, points } from "../data/format";
import type { ExperimentSummary, SiteStatus } from "../data/schema";
import { claimHeadline } from "../data/validate";

export function renderResultPanel(
  mount: HTMLElement,
  status: SiteStatus,
  summary: ExperimentSummary,
): void {
  if (status.phase !== "results" || !status.result_release) {
    mount.innerHTML = "";
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent =
      "Protocol only: no confirmatory result has been released yet.";
    mount.append(note);
    return;
  }

  const primary = summary.primary_summary;
  const budget = summary.budget.additional_revealed_targets.recovery;
  const panel = document.createElement("div");
  panel.className = "result-panel";
  panel.innerHTML = `
    <p class="headline">
      Recovery labels beat extra demonstrations by
      <span class="delta"></span>
      on the frozen unseen-corruption endpoint.
    </p>
    <p class="prose">
      <span data-slot="headline"></span>
      Mean paired difference <b data-slot="delta-text"></b>, 95% paired-<i>t</i>
      interval <b data-slot="interval"></b> across
      <b data-slot="bundles"></b> pipeline bundles, each evaluated on the same
      <b data-slot="eligible"></b> eligible unseen scenarios
      (intention-to-treat; +<span data-slot="budget"></span> labels per arm).
      Sensitivity (cluster bootstrap): <span data-slot="sensitivity"></span>.
      Smallest effect of interest: <span data-slot="sesoi"></span>.
    </p>
  `;
  const q = (name: string) =>
    panel.querySelector<HTMLElement>(`[data-slot="${name}"]`)!;
  panel.querySelector<HTMLElement>(".delta")!.textContent = points(
    primary.mean_paired_difference,
  );
  q("headline").textContent = claimHeadline(summary);
  q("delta-text").textContent = points(primary.mean_paired_difference, 2);
  q("interval").textContent = interval(
    primary.interval.lower,
    primary.interval.upper,
  );
  q("bundles").textContent = String(primary.pipeline_replicates);
  q("eligible").textContent = String(summary.eligibility.eligible_scenarios);
  q("budget").textContent = String(budget);
  q("sensitivity").textContent = interval(
    primary.sensitivity.lower,
    primary.sensitivity.upper,
  );
  q("sesoi").textContent = points(summary.protocol.sesoi_absolute_success)
    .replace("+", "±");
  mount.replaceChildren(panel);
}
