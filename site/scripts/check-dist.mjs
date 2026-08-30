// Post-build sanity: every asset the pages reference must exist in dist, and
// the entry bundles must fit the performance budget.

import { readFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { join, dirname, resolve } from "node:path";

const dist = resolve(process.cwd(), "dist");
if (!existsSync(dist)) {
  console.error("dist/ not found — run `npm run build` first");
  process.exit(1);
}

function walk(root) {
  const files = [];
  for (const entry of readdirSync(root)) {
    const path = join(root, entry);
    if (statSync(path).isDirectory()) files.push(...walk(path));
    else files.push(path);
  }
  return files;
}

const files = walk(dist);
const htmlFiles = files.filter((f) => f.endsWith(".html"));
let problems = 0;

// 1. Referenced local assets exist.
const attribute = /(?:src|href|poster)="([^"]+)"/g;
for (const page of htmlFiles) {
  const html = readFileSync(page, "utf-8");
  for (const match of html.matchAll(attribute)) {
    const url = match[1];
    if (/^(https?:|mailto:|#|data:)/.test(url)) continue;
    const clean = url.split(/[?#]/)[0];
    const target = clean.startsWith("/")
      ? join(dist, clean.replace(/^\/[^/]+\//, "/")) // strip the base segment
      : resolve(dirname(page), clean);
    const candidates = [
      target,
      join(target, "index.html"),
      // base-prefixed absolute paths also resolve from dist root
      join(dist, clean.replace(/^\//, "")),
    ];
    if (!candidates.some((c) => existsSync(c))) {
      console.error(`missing asset: ${url} (referenced by ${page})`);
      problems += 1;
    }
  }
}

// 1b. No build-time placeholder survived into the output.
for (const page of htmlFiles) {
  if (/%VITE_[A-Z_]+%/.test(readFileSync(page, "utf-8"))) {
    console.error(`unsubstituted build placeholder in ${page}`);
    problems += 1;
  }
}

// 2. Videos are referenced lazily (data-href), so check the staged media too.
for (const required of [
  "media/study/unseen_paired_contrast.mp4",
  "media/study/posters/unseen_paired_contrast.webp",
  "media/journey/recovery_contrast.mp4",
  "media/journey/trajectories/expert_labels.json",
  "media/journey/network/full_r0.json",
  "data/experiment-summary.json",
  "data/journey-data.json",
  "reports/Recovery_Policy_Learning_Technical_Report.pdf",
]) {
  if (!existsSync(join(dist, required))) {
    console.error(`missing staged file in dist: ${required}`);
    problems += 1;
  }
}

// 3. Budget: initial JS ≤ 100 kB gzip (hard ceiling 200), CSS ≤ 50 kB gzip.
const sizes = { js: 0, css: 0 };
for (const file of files) {
  if (file.includes("/assets/") && file.endsWith(".js")) {
    sizes.js += gzipSync(readFileSync(file)).length;
  }
  if (file.includes("/assets/") && file.endsWith(".css")) {
    sizes.css += gzipSync(readFileSync(file)).length;
  }
}
console.log(
  `bundle sizes (gzip): js ${(sizes.js / 1024).toFixed(1)} kB, ` +
    `css ${(sizes.css / 1024).toFixed(1)} kB`,
);
if (sizes.js > 200 * 1024) {
  console.error("JS exceeds the 200 kB hard ceiling");
  problems += 1;
} else if (sizes.js > 100 * 1024) {
  console.warn("JS exceeds the 100 kB target (still under the hard ceiling)");
}
if (sizes.css > 50 * 1024) {
  console.error("CSS exceeds the 50 kB budget");
  problems += 1;
}

if (problems > 0) {
  console.error(`check-dist: ${problems} problem(s)`);
  process.exit(1);
}
console.log(`check-dist: ok (${htmlFiles.length} pages, ${files.length} files)`);
