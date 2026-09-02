// Number formatting lives in one place so pages cannot improvise.

export function percent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function points(value: number, digits = 1): string {
  const scaled = value * 100;
  const sign = scaled > 0 ? "+" : "";
  return `${sign}${scaled.toFixed(digits)} pp`;
}

export function interval(lower: number, upper: number, digits = 1): string {
  return `[${points(lower, digits)}, ${points(upper, digits)}]`;
}

export function count(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatDate(iso: string): string {
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return iso;
  return parsed.toLocaleDateString("en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
  });
}

export function steps(value: number): string {
  return `${Math.round(value)} steps`;
}
