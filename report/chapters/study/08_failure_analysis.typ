#import "../../generated/metadata.typ": *
#import "../../generated/audit_result.typ": *
#import "../../components/format.typ": *

= Failure analysis <study-failures>

A success rate says how often a policy won. It says nothing about how it lost,
and in a closed-loop study that second question is often the more informative
one.

== There is only one way to fail here

Across the entire confirmatory opening, every failure is a step-limit
truncation. The count of episodes that terminated without reaching the goal is
#terminated-without-goal-total.

That is a degenerate composition, and reporting it is the point. In this
environment there is no lava, no irreversible action, and no way to end an
episode early except by succeeding: with the frozen three-action set and open
doors, an agent that has gone wrong simply keeps moving until the 144-step
limit expires. So "failure" throughout this report means one specific thing,
a policy that never recovered its way to the goal within the budget, and not a
mixture of failure modes that happens to average out.

This also means a whole class of question cannot be asked of this data. There
is no catastrophic-versus-recoverable distinction to measure, because nothing
here is catastrophic. A setting with irreversible failures would very likely
change the balance between the two arms, and that is one of the sharper limits
on transferring the result (@study-discussion).

#align(center)[
  #table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.header([*arm*], [*reached the goal*], [*hit the step limit*],
      [*ended without the goal*]),
    ..for row in failure-unseen {
      (
        [#arm-label.at(row.at(0))],
        [#count(row.at(1))],
        [#count(row.at(2))],
        [#count(row.at(3))],
      )
    },
  )
  #text(size: 8.5pt)[Terminal outcomes on the unseen slice, pooled over
    #bundles-completed bundles.]
]

#figure(
  image("../../generated/figures/failure_composition.png", width: 92%),
  caption: [Terminal outcome composition per arm on the unseen slice. The third
    category is empty by construction, not by omission.],
)

== Where the recovery arm still loses

#let recovery-outcomes = row-for(failure-unseen, "recovery_aggregation")
The recovery arm times out on #count(recovery-outcomes.at(2)) of
#count(recovery-outcomes.at(1) + recovery-outcomes.at(2)) assigned unseen
episodes. It is better than the alternatives, not close to solved: about one
assigned episode in
#dec((recovery-outcomes.at(1) + recovery-outcomes.at(2)) / recovery-outcomes.at(2),
     digits: 0)
still ends at the step limit.

The published media bundle therefore contains a recovery-arm *failure* as well
as a recovery-arm success, both selected by a disclosed ordinal rule rather
than by appearance: the paired contrast animation is the smallest eligible
unseen scenario where recovery succeeded and extra demonstrations failed, and
the failure animation is the smallest ordinal where the recovery arm failed
with a delivered corruption. Each replay is deterministic and its outcome is
asserted against the stored evaluation row before the file is written, so the
animation cannot drift from the evidence it illustrates.

== What the failures are not

They are not undelivered corruptions. Delivery was near universal on the unseen
slice (@study-results), and the handful of undelivered assignments remain in
every denominator, so they slightly *depress* every arm's rate rather than
inflating any of them.

They are not eligibility artifacts either. The eligibility filter removed
scenarios whose nominal oracle path was too short for the corruption to be
delivered at all, using a property of the scenario computed before any policy
existed, and it removed them identically for all three arms.
