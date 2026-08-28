#import "../../generated/metadata.typ": *
#import "../../generated/primary_result.typ": *
#import "../../generated/audit_result.typ": *
#import "../../generated/success_table.typ": *
#import "../../components/format.typ": *

= Results <study-results>

Everything in this chapter recomputes from the stored episode rows of the
single confirmatory opening. The primary endpoint comes first regardless of its
sign, followed by the secondary slices, the audits that make the comparison
checkable, and the costs that the design could not equalize.

== The primary endpoint

Across #bundles-completed complete pipeline bundles on the frozen eligible
unseen panel, with #eligible-count scenarios per cell and intention-to-treat
scoring, the mean paired difference in one-corruption success, recovery
aggregation minus extra demonstrations, is

$ hat(Delta) = #signed-pp(mean-delta, digits: 2) "pp", quad
  95% "paired" italic(t) " interval "
  #interval-pp(interval-lower, interval-upper, digits: 2) " pp". $

Applying the interpretation rule fixed at freeze time (@study-design) gives
claim state *#claim-state* with analysis status *#analysis-status*. The
interval lies entirely above the smallest effect size of interest, so the
result is not merely distinguishable from zero but larger than the threshold
declared worth caring about in advance.

#figure(
  image("../../generated/figures/primary_paired_effect.png", width: 88%),
  caption: [Primary paired contrast per pipeline bundle on the eligible unseen
    intention-to-treat slice. #eligible-count scenarios per cell,
    #bundles-completed bundles, mean with a 95% paired _t_ interval; the dashed
    line marks the smallest effect size of interest at #sesoi.],
)

#let positive-bundles = per-bundle-deltas.filter(row => row.at(1) > 0).len()
Every bundle contributes one point, and #positive-bundles of
#bundles-completed are positive:

#align(center)[
  #table(
    columns: (auto, auto),
    align: (left, right),
    table.header([*bundle*], [*paired difference (pp)*]),
    ..for (bundle, delta) in per-bundle-deltas {
      ([#bundle], [#signed-pp(delta, digits: 2)])
    },
  )
]

The crossed two-way cluster bootstrap, with #bootstrap-replicates replicates
and common bundle and scenario draws across arms, gives
#interval-pp(bootstrap-lower, bootstrap-upper, digits: 2) pp. It is a labelled
sensitivity analysis, not the primary interval, and it cannot compensate for
the small number of pipeline bundles, because it resamples the same
#bundles-completed trained pipelines. The prespecified precision target of a half-width at or below
#desired-half-width was
#if precision-met [met] else [not met], with an achieved half-width of
#dec(achieved-half-width, digits: 4).

== Success by slice and arm

#figure(
  image("../../generated/figures/success_matrix.png", width: 100%),
  caption: [Intention-to-treat success by slice and arm. Bars are means over
    #bundles-completed bundles, points are individual bundles, denominators are
    in the panel titles.],
)

#align(center)[
  #table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.header([*arm*], [*clean*], [*matched*], [*unseen (primary)*]),
    ..for arm in ("bc_base", "extra_demonstrations", "recovery_aggregation") {
      (
        [#arm-label.at(arm)],
        [#pct(row-for2(mean-success, arm, "clean").at(2))],
        [#pct(row-for2(mean-success, arm, "matched").at(2))],
        [#pct(row-for2(mean-success, arm, "unseen").at(2))],
      )
    },
  )
  #text(size: 8.5pt)[Mean over #bundles-completed bundles, #eligible-count
    assigned scenarios per cell, intention to treat.]
]

Two features of this table matter more than the headline.

First, the corruption bites. The base and extra-demonstration arms lose roughly
ten points when a held-out corruption is delivered, relative to their own clean
rates, while the recovery arm loses
#dec((row-for2(mean-success, "recovery_aggregation", "clean").at(2)
   - row-for2(mean-success, "recovery_aggregation", "unseen").at(2)) * 100)
points. That is the effect the study was designed to detect,
and it is visible without any statistics.

Second, the recovery arm also leads on the *clean* slice, where no corruption
is delivered at all. The benefit is therefore not purchased with a loss of
nominal performance, which is the trade-off one might have expected. It also
suggests the mechanism is broader than "the policy learned to undo this
operator", a reading taken up in @study-discussion.

== Secondary paired contrasts

#status-chip[#audit-status]

These contrasts use the same statistical unit as the primary endpoint and
differ only in which arms and which slice they compare. They were not
prespecified and are reported as secondary.

#align(center)[
  #table(
    columns: (auto, auto, auto, auto),
    align: (left, left, right, right),
    table.header([*contrast*], [*slice*], [*mean (pp)*], [*95% paired _t_ (pp)*]),
    ..for row in secondary-contrasts {
      (
        [#arm-short.at(row.at(0)) − #arm-short.at(row.at(1))],
        [#row.at(2)],
        [#signed-pp(row.at(3), digits: 2)],
        [#interval-pp(row.at(4), row.at(5), digits: 2)],
      )
    },
  )
]

#figure(
  image("../../generated/figures/secondary_contrasts.png", width: 92%),
  caption: [Secondary paired contrasts with every pipeline-bundle point.
    Diamonds are means with 95% paired _t_ intervals; the dashed line marks the
    smallest effect size of interest.],
)

The last row is the one a sceptical reader should look at first. The
extra-demonstration arm, the study's actual competitor, is not clearly
separated from the untouched base policy at this replicate count: its interval
against the base includes zero. Extra demonstrations plausibly helped, but with
#bundles-completed bundles the design cannot establish it. This does not weaken the primary
comparison, which is paired within bundles and far better resolved, but it does
bound how much should be read into the base line, and it is stated here rather
than left for someone else to compute.

== Assignment against delivery

Intention-to-treat scoring only means something if the gap between assignment
and delivery is visible. On the unseen slice, of #count(row-for2(delivery-rows,
"recovery_aggregation", "unseen").at(2)) assigned episodes per arm pooled over
bundles, the corruption failed to land in
#{ row-for2(delivery-rows, "bc_base", "unseen").at(4) },
#{ row-for2(delivery-rows, "extra_demonstrations", "unseen").at(4) }, and
#{ row-for2(delivery-rows, "recovery_aggregation", "unseen").at(4) } cases for
the base, extra-demonstration, and recovery arms respectively. On the matched
slice the counts are larger, because those corruptions are scheduled later.
Every one of these assignments stays in its denominator.

#figure(
  image("../../generated/figures/intervention_delivery.png", width: 88%),
  caption: [Assignments whose scheduled corruption never landed, pooled over
    bundles. Plotting delivered against assigned counts would show two nearly
    identical bars and hide the only quantity intention-to-treat accounting
    turns on.],
)

== Robustness across the corruption schedule

The unseen operator is scheduled at one of three times per scenario, fixed by
the contract. Splitting the primary slice by that time shows the advantage is
not an artifact of one particular moment of failure:

#align(center)[
  #{
    let times = time-profile.filter(r => r.at(0) == "bc_base").map(r => r.at(1))
    let header = ([*arm*],) + times.map(t => [*t = #t*])
    let body = ()
    for arm in ("bc_base", "extra_demonstrations", "recovery_aggregation") {
      body = body + ([#arm-label.at(arm)],)
      for row in time-profile.filter(r => r.at(0) == arm) {
        body = body + ([#pct(row.at(2) / row.at(3))],)
      }
    }
    table(
      columns: times.len() + 1,
      align: (left,) + times.map(_ => right),
      table.header(..header),
      ..body,
    )
  }
  #text(size: 8.5pt)[Success rate on the unseen slice by scheduled corruption
    time, pooled over #bundles-completed bundles.]
]

== The budget and exposure audit

#figure(
  image("../../generated/figures/budget_exposure.png", width: 100%),
  caption: [What the two full-budget arms share exactly, beside what they do
    not. Counts are summed over #bundles-completed bundles on a symmetric log
    scale, with exact values printed above each bar.],
)

The left panel is the claim of fairness made checkable. Base target exposures,
new target exposures, and optimizer updates are identical between the two arms,
recounted from the hash-chained ledgers rather than asserted from the
configuration.

The right panel is the cost the design deliberately did not equalize. Revealing
#budget-b recovery labels required
#count(row-for(budget-logged, "recovery_aggregation").at(1)) oracle calls and
#count(row-for(budget-logged, "recovery_aggregation").at(2)) simulator steps,
against #count(row-for(budget-logged, "extra_demonstrations").at(1)) for the same
number of nominal demonstration labels, and discarded
#count(row-for(budget-logged, "recovery_aggregation").at(3)) recommendations that
fell outside the reveal window. Recovery labels are therefore about
#dec(row-for(budget-logged, "recovery_aggregation").at(1)
     / row-for(budget-logged, "extra_demonstrations").at(1), digits: 1)
times more expensive to acquire in oracle queries. In a setting where the teacher is a scripted bot
this is bookkeeping; in a setting where the teacher is a person it would be the
dominant term, and any transfer of this result has to carry that number with it.

== Success-conditioned path cost

A method that succeeds more often can still take longer paths when it does
succeed, so the comparison is reported conditioned on success, with its
denominator.

#align(center)[
  #table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.header([*arm*], [*successes*], [*median step ratio*],
      [*mean step ratio*]),
    ..for row in overhead-unseen {
      (
        [#arm-label.at(row.at(0))],
        [#count(row.at(1)) / #count(row.at(2))],
        [#dec(row.at(3), digits: 3)],
        [#dec(row.at(4), digits: 3)],
      )
    },
  )
  #text(size: 8.5pt)[Steps taken relative to the nominal oracle path length, on
    the unseen slice, among successful episodes only.]
]

The recovery arm's successful episodes are modestly *longer* relative to the
oracle path than either comparison arm's. This is a real trade-off and it runs
against the headline: recovery buys a large gain in whether the task is solved
at a small cost in how directly it is solved. It is reported here, beside the
benefit, rather than in an appendix. The right panel of @fig-profile shows the
full distributions.

#figure(
  image("../../generated/figures/recovery_profile.png", width: 100%),
  caption: [Left: success by scheduled corruption time on the unseen slice.
    Right: path cost relative to the oracle among successful episodes only,
    with denominators in the panel title.],
) <fig-profile>

== Per-cell counts

The full crossed evaluation is #crossed-cells cells: #bundles-completed bundles
by three arms by three slices. Every cell is reported, as successes over
assigned episodes, with the number of delivered corruptions in parentheses.

#align(center)[
  #{
    let cell(bundle, arm, slice-name) = {
      let row = pipeline-rows.find(
        r => r.at(0) == bundle and r.at(1) == arm and r.at(2) == slice-name
      )
      if row == none {
        []
      } else if slice-name == "clean" {
        [#row.at(3) / #row.at(4)]
      } else {
        [#row.at(3) / #row.at(4) (#row.at(5))]
      }
    }
    let bundles = ()
    for row in pipeline-rows {
      if not bundles.contains(row.at(0)) { bundles.push(row.at(0)) }
    }
    let body = ()
    for bundle in bundles {
      for arm in ("bc_base", "extra_demonstrations", "recovery_aggregation") {
        body = body + (
          [#bundle], [#arm-short.at(arm)],
          cell(bundle, arm, "clean"),
          cell(bundle, arm, "matched"),
          cell(bundle, arm, "unseen"),
        )
      }
    }
    text(size: 8.5pt)[
      #table(
        columns: (auto, auto, auto, auto, auto),
        align: (left, left, right, right, right),
        table.header([*bundle*], [*arm*], [*clean*], [*matched*],
          [*unseen*]),
        ..body,
      )
    ]
  }
]
