// The idea in one moving picture: an expert's path, one corrupted action,
// the compounding drift it causes, and the two places labels could go.
// Geometry note: the learner's trajectory is authored as two path constants
// that share the golden dot's exact coordinates (C1-continuous through it),
// so the first reveal terminates at the dot by construction; it cannot
// overshoot. Its first control point sits on the expert path's own tangent at
// the branch point, so the dotted line leaves the corridor smoothly and only
// then peels away. Dashed strokes cannot be drawn-on directly, so each one is
// revealed through a mask holding a solid copy of the same geometry.

import { gsap } from "../../anim/gsap";
import { pointsAlong, svgEl } from "../../anim/helpers";
import type { StoryDefinition, StoryRefs } from "../story-player";

const CORRUPTION = { x: 505, y: 205 };
const EXPERT_D =
  "M 40 212 C 180 200, 300 178, 430 158 C 620 130, 820 100, 1000 80";
const LEARNER_1 = "M 410 161 C 474 151, 475 187, 505 205";
const LEARNER_2 =
  "M 505 205 C 530 220, 585 238, 640 232 S 730 205, 780 226 S 880 252, 930 240";

// Where the label dots sit, as arc-length fractions of their curves.
const EXTRA_FRACTIONS = [0.08, 0.22, 0.36, 0.5, 0.64, 0.78, 0.92];
const RECOVERY_FRACTIONS = [0.14, 0.3, 0.46, 0.62, 0.78, 0.92];

function buildStage(svg: SVGSVGElement): StoryRefs {
  const defs = svgEl("defs");
  const mask1 = svgEl("mask", { id: "drift-reveal-1" });
  const mask1Stroke = svgEl("path", {
    d: LEARNER_1, stroke: "#fff", "stroke-width": 10, fill: "none",
  });
  mask1.append(mask1Stroke);
  const mask2 = svgEl("mask", { id: "drift-reveal-2" });
  const mask2Stroke = svgEl("path", {
    d: LEARNER_2, stroke: "#fff", "stroke-width": 10, fill: "none",
  });
  mask2.append(mask2Stroke);
  defs.append(mask1, mask2);

  const corridor = svgEl("path", {
    d: EXPERT_D, stroke: "var(--extra-soft)", "stroke-width": 44,
    fill: "none", "stroke-linecap": "round",
  });
  const expert = svgEl("path", {
    d: EXPERT_D, stroke: "var(--extra)", "stroke-width": 3.5, fill: "none",
    "stroke-linecap": "round",
  });
  const expertLabel = svgEl("text", {
    x: 96, y: 164, fill: "var(--extra-strong)", "font-size": 15,
  });
  expertLabel.textContent = "the expert's demonstrations";

  const learner1 = svgEl("path", {
    d: LEARNER_1, stroke: "var(--caution)", "stroke-width": 3.5, fill: "none",
    "stroke-dasharray": "1 9", "stroke-linecap": "round",
    mask: "url(#drift-reveal-1)",
  });
  const learner2 = svgEl("path", {
    d: LEARNER_2, stroke: "var(--caution)", "stroke-width": 3.5, fill: "none",
    "stroke-dasharray": "1 9", "stroke-linecap": "round",
    mask: "url(#drift-reveal-2)",
  });
  const goldDot = svgEl("circle", {
    cx: CORRUPTION.x, cy: CORRUPTION.y, r: 8, fill: "var(--caution)",
  });
  const goldLabel = svgEl("text", {
    x: CORRUPTION.x + 14, y: CORRUPTION.y - 15, fill: "var(--caution-strong)",
    "font-size": 15,
  });
  goldLabel.textContent = "one corrupted action";
  const driftLabel = svgEl("text", {
    x: 690, y: 266, fill: "var(--ink-soft)", "font-size": 15,
  });
  driftLabel.textContent = "the learner's own drift";

  const groupExpert = svgEl("g");
  groupExpert.append(corridor, expert, expertLabel);
  const groupCorruption = svgEl("g");
  groupCorruption.append(learner1, goldDot, goldLabel);
  const groupDrift = svgEl("g");
  groupDrift.append(learner2, driftLabel);
  const groupLabels = svgEl("g");

  svg.append(defs, groupExpert, groupCorruption, groupDrift, groupLabels);

  // Label dots sit exactly on their curves, sampled by arc length.
  const extraDots: SVGElement[] = pointsAlong(
    expert as SVGPathElement, EXTRA_FRACTIONS,
  ).map((point) => {
    const dot = svgEl("circle", {
      cx: point.x, cy: point.y, r: 5.5, fill: "var(--extra)",
      stroke: "var(--paper-raised)", "stroke-width": 1.5,
    });
    groupLabels.append(dot);
    return dot;
  });
  const recoveryDots: SVGElement[] = [
    ...pointsAlong(learner1 as SVGPathElement, [0.78]),
    ...pointsAlong(learner2 as SVGPathElement, RECOVERY_FRACTIONS),
  ].map((point) => {
    const dot = svgEl("circle", {
      cx: point.x, cy: point.y, r: 5.5, fill: "var(--recovery)",
      stroke: "var(--paper-raised)", "stroke-width": 1.5,
    });
    groupLabels.append(dot);
    return dot;
  });

  const refs: StoryRefs = {
    corridor, expert, expertLabel, mask1Stroke, mask2Stroke,
    goldDot, goldLabel, driftLabel,
    groupExpert, groupCorruption, groupDrift, groupLabels,
  };
  extraDots.forEach((dot, i) => (refs[`extra${i}`] = dot));
  recoveryDots.forEach((dot, i) => (refs[`recovery${i}`] = dot));
  return refs;
}

function dots(refs: StoryRefs, prefix: string): SVGElement[] {
  return Object.keys(refs)
    .filter((key) => key.startsWith(prefix) && /\d$/.test(key))
    .sort((a, b) => Number(a.slice(prefix.length)) - Number(b.slice(prefix.length)))
    .map((key) => refs[key]!);
}

export const driftStory: StoryDefinition = {
  id: "drift",
  viewBox: "0 0 1040 280",
  ariaTitle: "How one corrupted action becomes compounding drift",
  ariaDesc:
    "An expert path crosses the scene. A dotted line peels away from it, " +
    "reaching a golden dot that marks one corrupted action, then wanders " +
    "further away. Finally, blue label dots appear along the expert path " +
    "and green label dots along the learner's drifted path.",
  buildStage,
  segments: [
    {
      id: "expert",
      title: "The expert's path",
      text:
        "A scripted expert solves the task, and its demonstrations trace a " +
        "narrow corridor of states. Behavioral cloning learns only here.",
      build: (refs) => {
        const tl = gsap.timeline();
        tl.fromTo(refs["corridor"]!, { opacity: 0 }, { opacity: 1, duration: 0.5 }, 0);
        tl.fromTo(
          refs["expert"]!,
          { drawSVG: "0%" },
          { drawSVG: "100%", duration: 1.3, ease: "power2.inOut" },
          0.1,
        );
        tl.fromTo(
          refs["expertLabel"]!,
          { opacity: 0 },
          { opacity: 1, duration: 0.4 },
          1.0,
        );
        return tl;
      },
    },
    {
      id: "corruption",
      title: "One corrupted action",
      text:
        "At deployment a single executed action is replaced. The very next " +
        "state already sits outside the corridor, and it was never labelled.",
      build: (refs) => {
        const tl = gsap.timeline();
        tl.fromTo(
          refs["goldDot"]!,
          { scale: 0, transformOrigin: "50% 50%" },
          { scale: 1, duration: 0.35, ease: "back.out(2)" },
          0,
        );
        tl.fromTo(
          refs["mask1Stroke"]!,
          { drawSVG: "0%" },
          { drawSVG: "100%", duration: 1.0, ease: "power2.inOut" },
          0.35,
        );
        tl.to(refs["goldDot"]!, { scale: 1.3, duration: 0.15, ease: "power1.in" }, 1.35);
        tl.to(refs["goldDot"]!, { scale: 1, duration: 0.2, ease: "power1.out" }, 1.5);
        tl.fromTo(
          refs["goldLabel"]!,
          { opacity: 0 },
          { opacity: 1, duration: 0.4 },
          1.1,
        );
        return tl;
      },
    },
    {
      id: "drift",
      title: "Errors compound",
      text:
        "From the unfamiliar state the policy keeps choosing actions it was " +
        "never taught, and each mistake feeds the next: the drift grows " +
        "on the order of T² over the horizon.",
      build: (refs) => {
        const tl = gsap.timeline();
        tl.fromTo(
          refs["mask2Stroke"]!,
          { drawSVG: "0%" },
          { drawSVG: "100%", duration: 1.8, ease: "power1.inOut" },
          0,
        );
        tl.fromTo(refs["driftLabel"]!, { opacity: 0 }, { opacity: 1, duration: 0.4 }, 1.4);
        return tl;
      },
    },
    {
      id: "labels",
      title: "Two ways to spend labels",
      text:
        "The budget question: more blue labels along the expert's corridor " +
        "(extra demonstrations), or green labels exactly on the states the " +
        "learner actually reached (recovery)?",
      build: (refs) => {
        const tl = gsap.timeline();
        tl.fromTo(
          dots(refs, "extra"),
          { scale: 0, transformOrigin: "50% 50%" },
          {
            scale: 1, duration: 0.35, ease: "back.out(2)",
            stagger: { amount: 0.6 },
          },
          0,
        );
        tl.fromTo(
          dots(refs, "recovery"),
          { scale: 0, transformOrigin: "50% 50%" },
          {
            scale: 1, duration: 0.35, ease: "back.out(2)",
            stagger: { amount: 0.6 },
          },
          0.8,
        );
        return tl;
      },
    },
  ],
  staticState: (refs, index) => {
    const groups = [
      refs["groupExpert"]!,
      refs["groupCorruption"]!,
      refs["groupDrift"]!,
      refs["groupLabels"]!,
    ];
    groups.forEach((group, i) => {
      (group as SVGElement).style.opacity = i <= index ? "1" : "0";
    });
  },
  captionVariants: {
    hero: {
      labels: {
        text:
          "The budget question this study answers: more blue labels along " +
          "the expert's corridor, or green labels exactly on the states the " +
          "learner reached after the mistake? The result is measured below.",
      },
    },
    chapter: {
      labels: {
        text:
          "More blue labels along the expert's corridor (extra " +
          "demonstrations), or green labels on the learner's own drifted " +
          "states (recovery)? This chapter measures both, in miniature.",
      },
    },
  },
};
