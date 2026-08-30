// Shared plumbing for the animated components: reduced-motion detection,
// viewport and tab-visibility hooks, path sampling, and a small SVG factory.

const SVG_NS = "http://www.w3.org/2000/svg";

/** Absent matchMedia (test environments) counts as reduced motion, so the
 *  static rendering path is the one exercised everywhere by default. */
export function prefersReducedMotion(): boolean {
  if (typeof window.matchMedia !== "function") return true;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

type VisibleOptions = {
  enter: () => void;
  exit?: () => void;
  threshold?: number;
  rootMargin?: string;
};

/** Run callbacks as an element enters/leaves the viewport; returns a dispose
 *  function. Without IntersectionObserver the element counts as visible. */
export function whenVisible(el: Element, opts: VisibleOptions): () => void {
  if (typeof IntersectionObserver === "undefined") {
    opts.enter();
    return () => {};
  }
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) opts.enter();
        else opts.exit?.();
      }
    },
    { threshold: opts.threshold ?? 0.3, rootMargin: opts.rootMargin ?? "0px" },
  );
  observer.observe(el);
  return () => observer.disconnect();
}

/** Pause work while the tab is hidden; returns a dispose function. */
export function whenPageVisible(opts: {
  hidden: () => void;
  visible: () => void;
}): () => void {
  const onChange = () => {
    if (document.hidden) opts.hidden();
    else opts.visible();
  };
  document.addEventListener("visibilitychange", onChange);
  return () => document.removeEventListener("visibilitychange", onChange);
}

/** Points at fractional arc lengths of a path. The SVG must be in the
 *  document; environments without geometry APIs get evenly spaced fallbacks
 *  along the path's bounding placement (fine for static test renders). */
export function pointsAlong(
  path: SVGPathElement,
  fractions: number[],
): Array<{ x: number; y: number }> {
  if (typeof path.getTotalLength !== "function") {
    return fractions.map((f) => ({ x: f * 100, y: 0 }));
  }
  const total = path.getTotalLength();
  return fractions.map((f) => {
    const point = path.getPointAtLength(total * f);
    return { x: point.x, y: point.y };
  });
}

/** Typed createElementNS with attributes. */
export function svgEl<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number> = {},
): SVGElementTagNameMap[K] {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attrs)) {
    el.setAttribute(name, String(value));
  }
  return el;
}

/** Resolve a design token to its hex value (single source: tokens.css). */
export function tokenColor(name: string): string {
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  return value || "#4f5f6b";
}

/** Piecewise-linear interpolation over the --value-0..4 ramp for a
 *  normalized value in [0, 1]. Stops are read once per call site. */
export function valueRamp(stops: string[]): (normalized: number) => string {
  const parsed = stops.map((hex) => {
    const clean = hex.replace("#", "");
    return [
      parseInt(clean.slice(0, 2), 16),
      parseInt(clean.slice(2, 4), 16),
      parseInt(clean.slice(4, 6), 16),
    ];
  });
  return (normalized: number) => {
    const clamped = Math.max(0, Math.min(1, normalized));
    const scaled = clamped * (parsed.length - 1);
    const low = Math.min(Math.floor(scaled), parsed.length - 2);
    const t = scaled - low;
    const a = parsed[low]!;
    const b = parsed[low + 1]!;
    const mix = (i: number) => Math.round(a[i]! + (b[i]! - a[i]!) * t);
    return `rgb(${mix(0)}, ${mix(1)}, ${mix(2)})`;
  };
}
