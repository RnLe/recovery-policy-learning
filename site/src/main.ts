// Landing page: the result exhibit plus the invitation into the journey.

import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/components.css";

import { markActiveNav } from "./components/chrome";
import { mountMedia } from "./components/media-player";
import { mountIllustrations } from "./components/placeholder";
import { renderResultPanel } from "./components/result-panel";
import { driftStory } from "./components/stories/drift-story";
import { mountStoryPlayer } from "./components/story-player";
import { renderThreeArm } from "./components/three-arm";
import { bindStats } from "./components/stats";
import { renderPairedEffect } from "./charts/paired-effect";
import { loadJourney, loadStudy, renderUnavailable } from "./data/load";

markActiveNav();
mountIllustrations();

const resultMount = document.querySelector<HTMLElement>(
  '[data-component="result-panel"]',
);

// The hero story needs no data, so it animates even if loading fails.
const storyMount = document.querySelector<HTMLElement>(
  '[data-component="drift-story"]',
);
if (storyMount) {
  mountStoryPlayer(storyMount, driftStory, {
    captionVariant: storyMount.dataset.storyVariant,
  });
}

void (async () => {
  try {
    const [{ status, summary }, journey] = await Promise.all([
      loadStudy(),
      loadJourney(),
    ]);
    bindStats({ summary, journey });
    if (resultMount) renderResultPanel(resultMount, status, summary);
    const effectMount = document.querySelector<HTMLElement>(
      '[data-component="paired-effect"]',
    );
    if (effectMount && status.result_release) {
      renderPairedEffect(effectMount, summary);
    }
    const armMount = document.querySelector<HTMLElement>(
      '[data-component="three-arm"]',
    );
    if (armMount) renderThreeArm(armMount, summary);
    mountMedia(journey.media);
  } catch (error) {
    console.error(error);
    if (resultMount) renderUnavailable(resultMount, error);
  }
})();
