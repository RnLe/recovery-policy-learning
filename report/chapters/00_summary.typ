#import "../generated/metadata.typ": *
#import "../generated/primary_result.typ": *
#import "../generated/audit_result.typ": *
#import "../components/format.typ": *

= Executive summary

A policy trained by behavioral cloning is fitted on the expert's states and
then, at test time, has to act under its own. The moment it makes a mistake it
is somewhere the demonstrations rarely went, and its error rate there is
unmeasured. The standard remedy is to spend expert labels in the states the
learner actually reaches. The standard alternative is simply to collect more
demonstrations. Which is the better use of the same budget is an accounting
question, and it is asked far less often than it is assumed.

This study asks it under strict matching. Given a fixed additional budget of
#budget-b revealed oracle labels, identical supervised target exposure, and
identical optimization, is that budget better spent on *recovery-state
aggregation*, meaning DAgger-style labels queried in states the learner visits
after one corrupted action, or on *extra nominal demonstrations* of the kind
the base policy already learned from?

The primary endpoint is eligible unseen one-corruption intention-to-treat
success: closed-loop task success on a frozen, never-touched panel of
#eligible-count scenarios, in which a held-out corruption operator replaces one
policy proposal per episode at a scheduled time. The estimate is the mean
within-bundle paired difference across #bundles-completed complete pipeline
replicates.

#align(center)[
  #box(inset: 9pt, stroke: 0.6pt, radius: 3pt)[
    #text(weight: "bold")[
      Recovery aggregation minus extra demonstrations =
      #signed-pp(mean-delta) percentage points
    ]
    #linebreak()
    95% paired _t_ interval
    #interval-pp(interval-lower, interval-upper) pp
    #linebreak()
    analysis status: #analysis-status · claim state: #emph(claim-state)
  ]
]

Three things about that number are worth stating immediately, because they are
what a reader should check rather than take on trust.

First, it was not a completion criterion. The frozen protocol committed in
advance to reporting a null, adverse, or inconclusive interval with exactly the
same prominence, and the interpretation rule was written down before the panel
was opened.

Second, the two arms really are matched. Both start from a bit-identical clone
of the same base checkpoint, reveal the same number of oracle targets, and take
the same number of optimizer updates on the same mix of old and new targets.
What they cannot share is the cost of acquisition, and that difference is
logged rather than called equal: the recovery arm made
#count(row-for(budget-logged, "recovery_aggregation").at(1)) oracle calls to
reveal its labels against
#count(row-for(budget-logged, "extra_demonstrations").at(1)) for extra
demonstrations, discarding
#count(row-for(budget-logged, "recovery_aggregation").at(3)) recommendations that
fell outside its reveal window.

Third, the advantage is not confined to the corrupted slice. Recovery labels
also beat extra demonstrations when no corruption is delivered at all, which
points at the on-policy character of the labels rather than at the policy
having learned to undo one specific operator.

The rest of Part I states the question precisely, describes what was built,
reports the primary endpoint with every replicate point, then reports the
trade-offs, the failures, two exploratory panels opened afterwards, and the
boundary of what this evidence can support.
