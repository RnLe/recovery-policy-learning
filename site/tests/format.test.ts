// @vitest-environment node
import { describe, expect, it } from "vitest";
import { count, formatDate, interval, percent, points } from "../src/data/format";

describe("formatting", () => {
  it("renders percentages and points", () => {
    expect(percent(0.143)).toBe("14.3%");
    expect(points(0.14303482587)).toBe("+14.3 pp");
    expect(points(-0.0133)).toBe("-1.3 pp");
    expect(points(0)).toBe("0.0 pp");
  });

  it("renders intervals in points", () => {
    expect(interval(0.10019895, 0.18587069)).toBe("[+10.0 pp, +18.6 pp]");
  });

  it("renders counts", () => {
    expect(count(28944)).toBe("28,944");
  });

  it("renders a readable date and leaves an unparseable one alone", () => {
    expect(formatDate("2026-08-26T11:12:40Z")).toBe("26 August 2026");
    expect(formatDate("not a date")).toBe("not a date");
  });
});
