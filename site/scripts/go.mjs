// Build the site the way it ships and serve it locally, in one step: refresh
// the staged evidence, build, run the dist check, start the preview server.
//
//   pnpm go [--base /recovery-policy-learning/] [--port 4173] [--host] [--open]

import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const siteDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(siteDir, "..");
const vite = resolve(siteDir, "node_modules/vite/bin/vite.js");

const argv = process.argv.slice(2);
const option = (name, fallback) => {
  const at = argv.indexOf(name);
  return at >= 0 && argv[at + 1] ? argv[at + 1] : fallback;
};
const base = option("--base", "/");
const port = option("--port", "4173");
const previewFlags = ["--host", "--open"].filter((flag) => argv.includes(flag));

// Ctrl-c is the normal way out. Before the server exists it just ends the run;
// afterwards it has to reach the server, or the port would stay held.
let preview = null;
const stop = (signal) => (preview ? preview.kill(signal) : process.exit(0));
process.on("SIGINT", () => stop("SIGINT"));
process.on("SIGTERM", () => stop("SIGTERM"));

// Steps run from the site directory unless they need the Python side.
// `optional` separates "tool is not installed" from "tool said no".
function run(command, args, { cwd = siteDir, optional = false } = {}) {
  const result = spawnSync(command, args, { cwd, stdio: "inherit" });
  if (result.error?.code === "ENOENT") {
    if (optional) return null;
    console.error(`${command} not found`);
    process.exit(1);
  }
  if (result.signal) process.exit(0); // interrupted, not failed
  return result.status ?? 0;
}

if (!existsSync(vite)) {
  const manager = (process.env.npm_config_user_agent ?? "").startsWith("pnpm")
    ? "pnpm"
    : "npm";
  console.log(`site dependencies missing — running ${manager} install`);
  const status = run(manager, ["install"]);
  if (status !== 0) process.exit(status);
}

// site/public is written only by `grf stage-site`; refresh it here rather than
// let the preview serve numbers that no longer match the evidence bundles.
const stagingManifest = resolve(siteDir, "public/staging-manifest.json");
if (!existsSync(stagingManifest)) {
  console.log("staging evidence into site/public");
  const status = run("uv", ["run", "grf", "stage-site"], {
    cwd: repoRoot,
    optional: true,
  });
  if (status === null) {
    console.error(
      "uv not found and nothing is staged yet — run `grf stage-site` first",
    );
    process.exit(1);
  }
  if (status !== 0) process.exit(status);
} else {
  const status = run("uv", ["run", "grf", "verify-staging"], {
    cwd: repoRoot,
    optional: true,
  });
  if (status === null) {
    console.warn("uv not found — serving the evidence already staged");
  } else if (status !== 0) {
    console.log("staged evidence no longer matches its sources — restaging");
    const restaged = run("uv", ["run", "grf", "stage-site", "--force"], {
      cwd: repoRoot,
    });
    if (restaged !== 0) process.exit(restaged);
  }
}

if (run(process.execPath, [vite, "build", "--base", base]) !== 0) {
  process.exit(1);
}
if (run(process.execPath, [resolve(siteDir, "scripts/check-dist.mjs")]) !== 0) {
  process.exit(1);
}

console.log(`\nproduction build at http://localhost:${port}${base}\n`);
preview = spawn(
  process.execPath,
  [
    vite,
    "preview",
    "--base",
    base,
    "--port",
    port,
    "--strictPort",
    ...previewFlags,
  ],
  { cwd: siteDir, stdio: "inherit" },
);
// vite reports the signal it stopped on (130 for ctrl-c, 143 for a kill).
// Stopping the server is how this command ends, so that is a clean exit.
preview.on("exit", (code, signal) => {
  const stopped = signal !== null || code === 130 || code === 143;
  process.exit(stopped ? 0 : (code ?? 0));
});
