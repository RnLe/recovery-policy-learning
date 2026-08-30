// Shared entry for the journey chapters. Pages declare mounts with
// data-component / data-stat / data-media / data-scrubber / data-illustration;
// this script fills them from the validated data.

import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/components.css";

import { renderBars } from "./charts/bars";
import { renderOutcomeMatrix } from "./charts/outcome-matrix";
import { renderPairedEffect } from "./charts/paired-effect";
import { markActiveNav } from "./components/chrome";
import { renderHashDemo } from "./components/hash-demo";
import { mountMedia } from "./components/media-player";
import { mountNetworkFlows } from "./components/network-flow";
import { mountIllustrations } from "./components/placeholder";
import { renderResultPanel } from "./components/result-panel";
import { mountScrubbers } from "./components/scrubber";
import { renderSliceExplainer } from "./components/slice-explainer";
import { bindStats } from "./components/stats";
import { driftStory } from "./components/stories/drift-story";
import { mountStoryPlayer } from "./components/story-player";
import { renderThreeArm } from "./components/three-arm";
import { loadJourney, loadStudy, renderUnavailable } from "./data/load";
import type { ExperimentSummary, JourneyData, SiteStatus } from "./data/schema";

markActiveNav();
mountIllustrations();

function mountCharts(
  journey: JourneyData,
  summary: ExperimentSummary,
  status: SiteStatus,
): void {
  for (const mount of document.querySelectorAll<HTMLElement>("[data-component]")) {
    switch (mount.dataset.component) {
      case "spectrum-bars": {
        const spectrum = journey.lab03.spectrum;
        renderBars(
          mount,
          [
            {
              label: "random policy",
              value: spectrum.random?.success_rate ?? 0,
              color: "var(--base-method)",
              detail: `${Math.round(spectrum.random?.mean_steps ?? 0)} steps on average`,
            },
            {
              label: "wall follower",
              value: spectrum.wall_follower?.success_rate ?? 0,
              color: "var(--caution)",
              detail: "hand-written right-hand rule",
            },
            {
              label: "scripted oracle",
              value: spectrum.oracle?.success_rate ?? 0,
              color: "var(--recovery)",
              detail: `${(spectrum.oracle?.mean_steps ?? 0).toFixed(1)} steps on average`,
            },
          ],
          {
            title: `Success on ${spectrum.oracle?.episodes ?? 0} identical scenarios`,
            description:
              "Success rates of the random policy, the wall follower, and the oracle.",
          },
        );
        break;
      }
      case "imitation-bars": {
        renderBars(
          mount,
          [
            {
              label: "memoryless",
              value: journey.lab04.memoryless.unseen_success_mean,
              color: "var(--base-method)",
              dots: journey.lab04.memoryless.unseen_success_per_seed,
              detail: "same data, no recurrent state",
            },
            {
              label: "recurrent (study model)",
              value: journey.lab04.recurrent.unseen_success_mean,
              color: "var(--recovery)",
              dots: journey.lab04.recurrent.unseen_success_per_seed,
              detail: "the architecture of the study",
            },
          ],
          {
            title: "Closed-loop success on unseen scenarios (3 seeds, dots)",
            description:
              "Mean unseen success of the memoryless and recurrent policies.",
          },
        );
        break;
      }
      case "ablation-bars": {
        renderBars(
          mount,
          journey.lab05.variants.map((variant) => ({
            label: variant.name.replace(/_/g, " "),
            value: variant.unseen_success_mean,
            color:
              variant.name === "full" ? "var(--recovery)" : "var(--base-method)",
            dots: variant.unseen_success_per_seed,
            detail: `${new Intl.NumberFormat("en-US").format(variant.parameters)} parameters`,
          })),
          {
            title: "One component removed at a time (3 seeds, dots)",
            description:
              "Unseen closed-loop success for each architecture variant.",
          },
        );
        break;
      }
      case "mini-arms-bars": {
        const matrix = journey.lab06.success_matrix;
        renderBars(
          mount,
          [
            {
              label: "base (+0 labels)",
              value: matrix.base?.unseen ?? 0,
              color: "var(--base-method)",
            },
            {
              label: "extra demos",
              value: matrix.extra?.unseen ?? 0,
              color: "var(--extra)",
            },
            {
              label: "recovery",
              value: matrix.recovery?.unseen ?? 0,
              color: "var(--recovery)",
            },
          ],
          {
            title: "Mini three-arm preview: unseen-corruption ITT success",
            description:
              "Unseen-slice success for the three arms of the small-scale preview.",
          },
        );
        break;
      }
      case "paired-effect":
        if (status.result_release) renderPairedEffect(mount, summary);
        break;
      case "outcome-matrix":
        if (status.result_release) renderOutcomeMatrix(mount, summary);
        break;
      case "three-arm":
        renderThreeArm(mount, summary);
        break;
      case "result-panel":
        renderResultPanel(mount, status, summary);
        break;
      case "drift-story":
        mountStoryPlayer(mount, driftStory, {
          captionVariant: mount.dataset.storyVariant,
        });
        break;
      case "slice-explainer":
        renderSliceExplainer(mount);
        break;
      case "hash-demo":
        renderHashDemo(mount);
        break;
      default:
        console.error(`unknown component: ${mount.dataset.component}`);
    }
  }
}

void (async () => {
  try {
    const [{ status, summary }, journey] = await Promise.all([
      loadStudy(),
      loadJourney(),
    ]);
    bindStats({ summary, journey });
    mountCharts(journey, summary, status);
    mountMedia(journey.media);
    await mountScrubbers();
    await mountNetworkFlows();
  } catch (error) {
    console.error(error);
    const main = document.querySelector<HTMLElement>("main .wrap");
    if (main) {
      const box = document.createElement("div");
      renderUnavailable(box, error);
      main.prepend(box);
    }
  }
})();
