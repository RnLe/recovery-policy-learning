// Reproduce page: chrome plus the handful of provenance numbers.

import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/components.css";

import { markActiveNav } from "./components/chrome";
import { bindStats } from "./components/stats";
import { loadJourney, loadStudy } from "./data/load";

markActiveNav();

void (async () => {
  try {
    const [{ summary }, journey] = await Promise.all([loadStudy(), loadJourney()]);
    bindStats({ summary, journey });
  } catch (error) {
    console.error(error);
  }
})();
