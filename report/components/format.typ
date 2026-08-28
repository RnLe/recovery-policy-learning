// Shared formatting helpers. Presentation only: no value is computed here,
// every number arrives from report/generated/.

#let arm-label = (
  bc_base: "BC base",
  extra_demonstrations: "Extra demonstrations",
  recovery_aggregation: "Recovery aggregation",
)

#let arm-short = (
  bc_base: "base",
  extra_demonstrations: "extra",
  recovery_aggregation: "recovery",
)

// A number at a fixed number of decimals. Typst drops trailing zeros, which
// makes 10.0 and 10 look like different precisions in the same interval.
#let dec(x, digits: 1) = {
  let text-body = str(calc.round(x, digits: digits))
  if digits > 0 {
    let parts = text-body.split(".")
    let frac = if parts.len() > 1 { parts.at(1) } else { "" }
    if frac.len() < digits {
      text-body = (
        parts.at(0) + "." + frac + "0" * (digits - frac.len())
      )
    }
  }
  text-body
}

// An integer with thousands separators, for counts a reader has to compare.
#let count(n) = {
  let digits-of = str(calc.abs(int(n)))
  let grouped = ()
  let rest = digits-of
  while rest.len() > 3 {
    grouped.push(rest.slice(rest.len() - 3))
    rest = rest.slice(0, rest.len() - 3)
  }
  grouped.push(rest)
  let joined = grouped.rev().join(",")
  if int(n) < 0 { "-" + joined } else { joined }
}

// A rate as a percentage.
#let pct(x, digits: 1) = dec(x * 100, digits: digits) + "%"

// A difference in percentage points, always carrying its sign.
#let signed-pp(x, digits: 1) = {
  let v = calc.round(x * 100, digits: digits)
  if v >= 0 { "+" + dec(v, digits: digits) } else { dec(v, digits: digits) }
}

// An interval in percentage points.
#let interval-pp(lower, upper, digits: 1) = (
  "[" + signed-pp(lower, digits: digits) + ", "
      + signed-pp(upper, digits: digits) + "]"
)

// First row of `rows` whose column `index` equals `key`.
#let row-for(rows, key, index: 0) = rows.find(r => r.at(index) == key)

// First row matching two columns.
#let row-for2(rows, first, second) = rows.find(
  r => r.at(0) == first and r.at(1) == second
)

// A small status chip, used to keep evidence labels visible next to results.
#let status-chip(text-body, fill: rgb("#eef3f6")) = box(
  fill: fill, inset: (x: 5pt, y: 2.5pt), radius: 2pt,
)[#text(size: 7.5pt, weight: "bold")[#upper(text-body)]]

// A callout for the boundary of what a result can support.
#let scope-note(body) = block(
  fill: rgb("#fdf3e3"), inset: 8pt, radius: 3pt, width: 100%,
)[#body]
