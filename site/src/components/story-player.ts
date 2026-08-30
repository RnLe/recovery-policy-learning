// A small story stage: one SVG scene told in timed sections, each with its own
// progress bar and explanation shown side by side underneath. The story loops
// continuously; the section in focus is in full ink and the others are greyed.
// Clicking a section jumps there. Stories only define their stage and
// per-section timelines; playback, focus, and reduced motion live here once.

import { gsap, type Timeline } from "../anim/gsap";
import {
  prefersReducedMotion,
  whenPageVisible,
  whenVisible,
} from "../anim/helpers";

export type StoryRefs = Record<string, SVGElement>;

export type StorySegment = {
  id: string;
  title: string;
  text: string;
  /** Build this section's animation over the shared stage. The returned
   *  timeline's own duration is the animation length. */
  build: (refs: StoryRefs) => Timeline;
};

export type StoryDefinition = {
  id: string;
  viewBox: string;
  ariaTitle: string;
  ariaDesc: string;
  /** Create every SVG element once, initial states authored as attributes. */
  buildStage: (svg: SVGSVGElement) => StoryRefs;
  segments: StorySegment[];
  /** Render section ``index`` finished, without any timeline, so the whole
   *  story under reduced motion (and in test environments without SVG
   *  geometry APIs, where the animation plugins cannot run). */
  staticState: (refs: StoryRefs, index: number) => void;
  /** Optional per-mount caption overrides, keyed by variant then segment id. */
  captionVariants?: Record<
    string,
    Record<string, { title?: string; text?: string }>
  >;
};

export type StoryPlayerHandle = {
  jumpTo(segment: number): void;
  play(): void;
  pause(): void;
  destroy(): void;
};

// Each section is displayed for HOLD_FACTOR times its animation length: the
// animation plays, then the finished scene holds while the bar keeps filling.
const HOLD_FACTOR = 2;
const MIN_DISPLAY_SECONDS = 2.5;
const LOOP_PAUSE_SECONDS = 1.5; // a breath on the final scene before looping

export function mountStoryPlayer(
  mount: HTMLElement,
  story: StoryDefinition,
  options: { captionVariant?: string } = {},
): StoryPlayerHandle {
  const reduced = prefersReducedMotion();

  mount.className = "story";
  mount.innerHTML = `
    <svg class="story__stage" viewBox="${story.viewBox}" role="img">
      <title>${story.ariaTitle}</title>
      <desc>${story.ariaDesc}</desc>
    </svg>
    <div class="story__steps" role="group" aria-label="story sections"></div>
  `;
  const svg = mount.querySelector<SVGSVGElement>("svg")!;
  const stepRow = mount.querySelector<HTMLElement>(".story__steps")!;

  const refs = story.buildStage(svg);

  const overrides = story.captionVariants?.[options.captionVariant ?? ""] ?? {};
  const captionFor = (index: number) => {
    const segment = story.segments[index]!;
    const override = overrides[segment.id] ?? {};
    return {
      title: override.title ?? segment.title,
      text: override.text ?? segment.text,
    };
  };

  // All sections are on the page at once; the active one is highlighted.
  const steps: HTMLButtonElement[] = [];
  const fills: HTMLElement[] = [];
  story.segments.forEach((segment, index) => {
    const { title, text } = captionFor(index);
    const step = document.createElement("button");
    step.type = "button";
    step.className = "story__step";
    // No aria-label: the title and explanation are the accessible name, so a
    // screen reader hears exactly the words on the page.
    step.innerHTML =
      `<span class="story__bar" aria-hidden="true">` +
      `<span class="story__bar-fill"></span></span>` +
      `<b class="story__step-title">${title}.</b> ` +
      `<span class="story__step-text">${text}</span>`;
    stepRow.append(step);
    steps.push(step);
    fills.push(step.querySelector<HTMLElement>(".story__bar-fill")!);
    step.addEventListener("click", () => jumpTo(index));
  });

  let active = -1;
  const setActive = (index: number) => {
    if (index === active) return;
    active = index;
    steps.forEach((step, i) => {
      if (i === index) step.setAttribute("aria-current", "step");
      else step.removeAttribute("aria-current");
    });
  };

  // One master timeline (motion mode only): each section's animation plus a
  // bar-fill tween spanning its whole display window, so seeking keeps the
  // scene and the highlighted section honest together.
  const master = reduced
    ? null
    : gsap.timeline({
        paused: true,
        repeat: -1,
        repeatDelay: LOOP_PAUSE_SECONDS,
      });
  const starts: number[] = [];
  let cursor = 0;
  if (master) {
    story.segments.forEach((segment, index) => {
      starts.push(cursor);
      const section = segment.build(refs); // reparented into the master
      master.add(section, cursor);
      const display = Math.max(
        section.duration() * HOLD_FACTOR,
        MIN_DISPLAY_SECONDS,
      );
      master.set(fills[index]!, { scaleX: 0 }, cursor);
      master.to(
        fills[index]!,
        { scaleX: 1, duration: display, ease: "none" },
        cursor,
      );
      cursor += display;
    });
    // The active section is derived from the playhead rather than marked with
    // callbacks: seek() suppresses those by design, and this stays correct
    // across jumps and loops alike.
    const indexAt = (time: number) => {
      let index = 0;
      while (index + 1 < starts.length && time >= starts[index + 1]!) {
        index += 1;
      }
      return index;
    };
    master.eventCallback("onUpdate", () => setActive(indexAt(master.time())));
    master.eventCallback("onRepeat", () => setActive(0));
  }

  let inView = true;
  const play = () => {
    master?.play();
  };
  const pause = () => {
    master?.pause();
  };
  const jumpTo = (index: number) => {
    setActive(index);
    if (!master) {
      story.staticState(refs, index);
      fills.forEach((fill, i) => {
        fill.style.transform = `scaleX(${i <= index ? 1 : 0})`;
      });
      return;
    }
    master.seek(starts[index]!);
    play();
  };

  const disposers: Array<() => void> = [];
  if (reduced) {
    // Static mode: show each section's finished state on demand.
    jumpTo(0);
  } else {
    setActive(0);
    disposers.push(
      whenVisible(mount, {
        threshold: 0.35,
        enter: () => {
          inView = true;
          play();
        },
        exit: () => {
          inView = false;
          pause();
        },
      }),
      whenPageVisible({
        hidden: () => pause(),
        visible: () => {
          if (inView) play();
        },
      }),
    );
  }

  return {
    jumpTo,
    play,
    pause,
    destroy: () => {
      master?.kill();
      disposers.forEach((dispose) => dispose());
    },
  };
}
