// @vitest-environment node
// Every data-stat key used by any page must resolve in the registry; a typo
// in HTML fails the suite instead of rendering an em dash in production.

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { STATS } from "../src/components/stats";

function htmlFiles(root: string): string[] {
  const found: string[] = [];
  for (const entry of readdirSync(root)) {
    if (["node_modules", "dist", "public"].includes(entry)) continue;
    const path = join(root, entry);
    if (statSync(path).isDirectory()) {
      found.push(...htmlFiles(path));
    } else if (entry.endsWith(".html")) {
      found.push(path);
    }
  }
  return found;
}

describe("stat coverage", () => {
  const pages = htmlFiles(new URL("..", import.meta.url).pathname);

  it("finds the pages", () => {
    expect(pages.length).toBeGreaterThanOrEqual(10);
  });

  it("every data-stat key in HTML exists in the registry", () => {
    const used = new Set<string>();
    for (const page of pages) {
      const html = readFileSync(page, "utf-8");
      for (const match of html.matchAll(/data-stat="([^"]+)"/g)) {
        used.add(match[1]!);
      }
    }
    expect(used.size).toBeGreaterThan(20);
    const unknown = [...used].filter((key) => !(key in STATS));
    expect(unknown).toEqual([]);
  });

  it("every media id in HTML is a known study or journey clip", () => {
    const known = new Set([
      "unseen_paired_contrast",
      "unseen_recovery_failure",
      "expert_labels",
      "random_wander",
      "aliasing_pair",
      "imitation_contrast",
      "recovery_contrast",
      "recovery_failure",
    ]);
    for (const page of pages) {
      const html = readFileSync(page, "utf-8");
      for (const match of html.matchAll(/data-media="([^"]+)"/g)) {
        expect(known.has(match[1]!), `${match[1]} in ${page}`).toBe(true);
      }
    }
  });
});
