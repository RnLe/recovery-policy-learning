// @vitest-environment node
import { afterEach, describe, expect, it, vi } from "vitest";

describe("publicUrl under a project-pages base", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  it("prefixes the configured base and strips leading slashes", async () => {
    vi.stubEnv("BASE_URL", "/recovery-policy-learning/");
    const { publicUrl } = await import("../src/data/paths");
    expect(publicUrl("data/site-status.json")).toBe(
      "/recovery-policy-learning/data/site-status.json",
    );
    expect(publicUrl("/media/study/unseen_paired_contrast.mp4")).toBe(
      "/recovery-policy-learning/media/study/unseen_paired_contrast.mp4",
    );
  });

  it("works at the dev-server root", async () => {
    vi.stubEnv("BASE_URL", "/");
    const { publicUrl } = await import("../src/data/paths");
    expect(publicUrl("data/journey-data.json")).toBe("/data/journey-data.json");
  });
});
