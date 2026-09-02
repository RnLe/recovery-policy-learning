// Fetch, validate, cache. Pages ask for data; they never fetch directly.

import type { ExperimentSummary, JourneyData, SiteStatus } from "./schema";
import { publicUrl } from "./paths";
import { validateStatus, validateSummary } from "./validate";

export class DataError extends Error {
  readonly problems: string[];

  constructor(message: string, problems: string[] = []) {
    super(message);
    this.problems = problems;
  }
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(publicUrl(path));
  if (!response.ok) {
    throw new DataError(`${path}: HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

let studyCache: Promise<{
  status: SiteStatus;
  summary: ExperimentSummary;
}> | null = null;

export function loadStudy() {
  studyCache ??= (async () => {
    const status = await fetchJson<SiteStatus>("data/site-status.json");
    const statusProblems = validateStatus(status);
    if (statusProblems.length > 0) {
      throw new DataError("site status failed validation", statusProblems);
    }
    const summary = await fetchJson<ExperimentSummary>(
      "data/experiment-summary.json",
    );
    const problems = validateSummary(summary, status);
    if (problems.length > 0) {
      throw new DataError("experiment summary failed validation", problems);
    }
    return { status, summary };
  })();
  return studyCache;
}

let journeyCache: Promise<JourneyData> | null = null;

export function loadJourney(): Promise<JourneyData> {
  journeyCache ??= fetchJson<JourneyData>("data/journey-data.json");
  return journeyCache;
}

export function renderUnavailable(mount: HTMLElement, error: unknown): void {
  const box = document.createElement("div");
  box.className = "result-unavailable";
  const detail =
    error instanceof DataError && error.problems.length > 0
      ? ` (${error.problems.join("; ")})`
      : "";
  box.textContent = `results unavailable: validation failed${detail}`;
  mount.replaceChildren(box);
}
