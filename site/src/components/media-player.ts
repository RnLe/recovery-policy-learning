// Rollout videos: poster first, source attached near the viewport, then
// muted autoplay in view and paused out of it. Native controls stay off; an
// overlay appears on hover or keyboard focus with play/pause and true
// frame stepping (the clips are constant-rate, so one frame = 1/fps).
// Each caption reads as a sentence: the mission, how the episode went, and
// the rule that picked this scenario out of the evaluated set.

import { prefersReducedMotion, whenVisible } from "../anim/helpers";
import { publicUrl } from "../data/paths";
import type { MediaItem, MediaManifest } from "../data/schema";

const STUDY_MEDIA: Record<string, { href: string; poster: string }> = {
  unseen_paired_contrast: {
    href: "media/study/unseen_paired_contrast.mp4",
    poster: "media/study/posters/unseen_paired_contrast.webp",
  },
  unseen_recovery_failure: {
    href: "media/study/unseen_recovery_failure.mp4",
    poster: "media/study/posters/unseen_recovery_failure.webp",
  },
};

function describe(item: MediaItem): string {
  const ran =
    item.outcome === "success"
      ? item.steps
        ? `solved in ${item.steps} steps`
        : "solved"
      : item.outcome === "failure"
        ? item.steps
          ? `${item.steps} steps, and the goal stays out of reach`
          : "the goal stays out of reach"
        : item.outcome;
  const mission = item.mission ? `“${item.mission}”, ${ran}. ` : `${ran}. `;
  return `${mission}${item.selection_rule}`;
}

function attach(video: HTMLVideoElement, href: string): void {
  if (video.dataset.attached) return;
  video.dataset.attached = "true";
  const source = document.createElement("source");
  source.src = publicUrl(href);
  source.type = "video/mp4";
  video.append(source);
  video.load();
}

function addControls(
  video: HTMLVideoElement,
  fps: number,
  reduced: boolean,
): void {
  const frame = 1 / fps;
  const shell = document.createElement("div");
  shell.className = "player";
  video.replaceWith(shell);
  shell.append(video);

  const overlay = document.createElement("div");
  overlay.className = "player__controls";
  overlay.innerHTML = `
    <button type="button" data-frame="-1" aria-label="one frame back">‹</button>
    <button type="button" data-play aria-label="pause">❚❚</button>
    <button type="button" data-frame="1" aria-label="one frame forward">›</button>
    <span class="player__time" aria-hidden="true"></span>
  `;
  shell.append(overlay);
  if (reduced) shell.classList.add("player--static");

  const playButton = overlay.querySelector<HTMLButtonElement>("[data-play]")!;
  const time = overlay.querySelector<HTMLElement>(".player__time")!;

  const showPlaying = (playing: boolean) => {
    playButton.textContent = playing ? "❚❚" : "▶";
    playButton.setAttribute("aria-label", playing ? "pause" : "play");
  };
  video.addEventListener("play", () => showPlaying(true));
  video.addEventListener("pause", () => showPlaying(false));
  video.addEventListener("timeupdate", () => {
    time.textContent = `frame ${Math.round(video.currentTime * fps)}`;
  });
  showPlaying(false);

  playButton.addEventListener("click", () => {
    if (video.paused) void video.play();
    else video.pause();
  });
  for (const button of overlay.querySelectorAll<HTMLButtonElement>(
    "[data-frame]",
  )) {
    button.addEventListener("click", () => {
      video.pause();
      const direction = Number(button.dataset.frame);
      const duration = Number.isFinite(video.duration) ? video.duration : 0;
      let next = video.currentTime + direction * frame;
      if (duration > 0) {
        // wrap so stepping keeps working at both ends of a looping clip
        if (next < 0) next = duration - frame / 2;
        if (next >= duration) next = 0;
      }
      video.currentTime = Math.max(0, next);
    });
  }
}

export function mountMedia(manifest: MediaManifest): void {
  const reduced = prefersReducedMotion();
  const fps = manifest.fps || 5;
  const items = new Map<string, MediaItem>(
    manifest.items.map((item) => [item.id, item]),
  );

  for (const video of document.querySelectorAll<HTMLVideoElement>(
    "video[data-media]",
  )) {
    const id = video.dataset.media ?? "";
    const journeyItem = items.get(id);
    const study = STUDY_MEDIA[id];
    const href = journeyItem?.href ?? study?.href;
    const poster = journeyItem?.poster ?? study?.poster;
    if (!href || !poster) {
      console.error(`unknown media id: ${id}`);
      continue;
    }
    video.poster = publicUrl(poster);
    video.controls = false;
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = "metadata";
    video.dataset.href = href;
    addControls(video, fps, reduced);

    // Attach near the viewport; autoplay while visible unless the user
    // prefers reduced motion or explicitly paused this clip.
    whenVisible(video, {
      rootMargin: "300px",
      threshold: 0.01,
      enter: () => attach(video, href),
    });
    if (!reduced) {
      let userPaused = false;
      video.addEventListener("pause", () => {
        if (!video.dataset.autopause) userPaused = true;
        delete video.dataset.autopause;
      });
      video.addEventListener("play", () => {
        userPaused = false;
      });
      whenVisible(video, {
        threshold: 0.4,
        enter: () => {
          if (!userPaused) void video.play().catch(() => {});
        },
        exit: () => {
          if (!video.paused) {
            video.dataset.autopause = "true";
            video.pause();
          }
        },
      });
    }

    const caption = video
      .closest("figure")
      ?.querySelector<HTMLElement>("[data-media-caption]");
    if (caption && journeyItem) {
      caption.textContent = describe(journeyItem);
    }
  }
}
