import { readFileSync } from "node:fs";

import { defineConfig } from "vite";

// The footer's "last updated" is the day the site was built, formatted the way
// formatDate() renders every other date on the site. Vite substitutes it into
// %VITE_BUILD_DATE% in each HTML entry, in dev and in the build alike.
process.env.VITE_BUILD_DATE = new Date().toLocaleDateString("en-GB", {
  day: "numeric",
  month: "long",
  year: "numeric",
  timeZone: "UTC",
});

// The report link carries the size of the file a reader is about to open,
// read from the staged PDF itself. The page tree is uncompressed, so the
// largest /Count in the file is the page total, no PDF library needed.
const reportPdf = new URL(
  "public/reports/Recovery_Policy_Learning_Technical_Report.pdf",
  import.meta.url,
).pathname;
process.env.VITE_REPORT_META = (() => {
  try {
    const bytes = readFileSync(reportPdf);
    const counts = [...bytes.toString("latin1").matchAll(/\/Count\s+(\d+)/g)].map(
      (match) => Number(match[1]),
    );
    if (counts.length === 0) return "";
    return `(${Math.max(...counts)}p, ${(bytes.length / 1e6).toFixed(1)} MB)`;
  } catch {
    return ""; // staging has not run yet; the link still works
  }
})();

// Production builds pass the Pages base on the command line, e.g.
//   npm run build -- --base "/recovery-policy-learning/"
// so local dev keeps the plain "/" base.
export default defineConfig({
  build: {
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      input: {
        index: new URL("index.html", import.meta.url).pathname,
        world: new URL("journey/01-world/index.html", import.meta.url).pathname,
        decision: new URL("journey/02-decision/index.html", import.meta.url).pathname,
        oracle: new URL("journey/03-oracle/index.html", import.meta.url).pathname,
        learning: new URL("journey/04-learning/index.html", import.meta.url).pathname,
        architecture: new URL("journey/05-architecture/index.html", import.meta.url)
          .pathname,
        shift: new URL("journey/06-shift/index.html", import.meta.url).pathname,
        measurement: new URL("journey/07-measurement/index.html", import.meta.url)
          .pathname,
        study: new URL("journey/08-study/index.html", import.meta.url).pathname,
        reproduce: new URL("reproduce/index.html", import.meta.url).pathname,
      },
    },
  },
});
