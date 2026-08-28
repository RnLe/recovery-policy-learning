#import "../../generated/metadata.typ": *
#import "../../generated/exploratory_result.typ": *
#import "../../components/format.typ": *

= Exploratory extensions <study-exploratory>

#status-chip[#exploratory-status]

Eight scenario splits were generated, hashed, and frozen before the study ran.
Six of them carried the experiment. The remaining two, `expert_diagnostic` and
`difficulty_shift`, were reserved for optional descriptive diagnostics and were
not part of the confirmatory opening.

They are opened here, after the confirmatory result, and everything in this
chapter is exploratory: not prespecified, not part of any claim, and recorded
with its own opening note so that a post-hoc opening is itself on the record.
Neither panel can move the primary estimand, and neither is used to support it.

== Expert agreement is not closed-loop success

Each arm's final policy was run closed loop with no corruption on
#agreement-scenarios scenarios per bundle, while the scripted oracle ran beside
it in lockstep. At every step the policy's greedy action was compared with the
oracle's recommendation for that same state.

#align(center)[
  #table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    table.header([*arm*], [*agrees with the oracle, per step*],
      [*reaches the goal, per episode*]),
    ..for row in agreement-rows {
      ([#arm-label.at(row.at(0))], [#pct(row.at(1))], [#pct(row.at(2))])
    },
  )
]

#figure(
  image("../../generated/figures/exploratory_expert_agreement.png", width: 92%),
  caption: [Per-step agreement with the scripted oracle beside per-episode
    closed-loop success, with no corruption delivered.],
)

The gap between the two columns is the finding. The base policy matches the
oracle's exact action on fewer than half of its steps and still reaches the
goal four times in five. Agreement is a poor proxy for the thing the study
actually measures, because in an open maze many actions are equally valid: two
routes around a wall differ at every step and arrive at the same place.

This is the concrete reason the primary endpoint is closed-loop task success
rather than action accuracy. A study that optimized or reported agreement would
be measuring conformity to one particular expert trajectory, not competence.

The ordering is nonetheless worth noting: the recovery arm agrees with the
oracle substantially more often, which is consistent with its labels having
come from states the policy itself reaches. It is a descriptive observation
here, with no interval and no claim attached.

== Two corruptions instead of one

The primary endpoint is explicitly a one-corruption endpoint. This panel
schedules *two* held-out corruptions per episode, at distinct times drawn from
the unseen time set, over #two-corruption-scenarios scenarios per cell, and
asks whether the advantage survives compounding.

#align(center)[
  #table(
    columns: (auto, auto),
    align: (left, right),
    table.header([*arm*], [*ITT success, two corruptions*]),
    ..for row in two-corruption-rows {
      ([#arm-label.at(row.at(0))], [#pct(row.at(1))])
    },
  )
]

#figure(
  image("../../generated/figures/exploratory_two_corruption.png", width: 88%),
  caption: [Intention-to-treat success under two held-out corruptions per
    episode. Bars are means over #bundles-completed bundles, points are
    individual bundles.],
)

The paired recovery-minus-extra difference is
#signed-pp(two-corruption-mean, digits: 1) pp
#interval-pp(two-corruption-lower, two-corruption-upper, digits: 1) pp,
computed on the same statistical unit as the primary endpoint. The advantage
persists under a harder perturbation, and is somewhat smaller than under a
single corruption.

Scoring is intention to treat here as well, and it matters more on this panel
than on the confirmatory one: roughly one episode in eight ends before its
second scheduled corruption can be delivered. Those assignments stay in the
denominator, and the per-cell delivered counts are in
`exploratory/tables/two_corruption.csv`.

Read this as a direction rather than a measurement. The panel is a fifth the
size of the confirmatory one, it was opened after the result was known, and its
interval is descriptive. What it does rule out is one specific worry, that the
recovery arm's advantage depends on the perturbation being exactly the
single-corruption event it was evaluated under.
