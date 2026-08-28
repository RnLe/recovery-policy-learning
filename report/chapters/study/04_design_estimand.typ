#import "../../generated/metadata.typ": *
#import "../../generated/audit_result.typ": *
#import "../../components/format.typ": *

= Study design <study-design>

The design principles used here, meaning matched budgets, intention-to-treat
scoring, and paired replication, are each demonstrated with known-truth
simulations in @lab-measurement; the three-arm structure is previewed at small
scale in @lab-shift.

== Three arms, one shared start

Each pipeline bundle trains one base policy on the shared dataset $D_0$ of
exactly #n0 revealed targets, then clones it, parameters and optimizer state
asserted bit-identical, into the two full-budget arms:

- *bc_base*: the untouched base checkpoint, zero post-base updates. A context
  line, not the competitor.
- *extra demonstrations*: #budget-b additional oracle targets drawn from fresh
  nominal trajectories, #{ calc.quo(budget-b, rounds-k) } per round over
  #rounds-k rounds.
- *recovery aggregation*: #budget-b additional oracle targets drawn from the
  learner's own post-corruption states. The current policy rolls out, one
  scheduled proposal is corrupted to $g_+($proposal$)$ and executed, and the
  oracle's recommendations for at most #window-h successive states after the
  corrupted transition are revealed, until the exact per-round budget is met.

Both arms are re-cloned from the *same* base checkpoint in every bundle, so a
bundle contributes one paired difference in which the only thing that differs
is where the labels came from.

== What is matched, and what is only logged

After each round both arms train for exactly the same number of optimizer
updates with identical base and new target slots per update. A fairness audit
recounts the immutable ledgers and refuses to continue on any inequality in
updates, drawn targets, or cumulative exposures. Over the #bundles-completed bundles that
comes to #count(row-for(budget-matched, "extra_demonstrations").at(1)) base target
exposures, #count(row-for(budget-matched, "extra_demonstrations").at(2)) new
target exposures, and
#count(row-for(budget-matched, "extra_demonstrations").at(3)) optimizer updates
for each of the two arms, identical on every count.

Quantities that cannot be matched are logged and reported rather than called
equal. Acquiring recovery labels means running the learner, so most of the
oracle recommendations produced along the way fall outside the reveal window
and are discarded. Over the #bundles-completed bundles the extra-demonstration arm made
#count(row-for(budget-logged, "extra_demonstrations").at(1)) oracle calls and
took #count(row-for(budget-logged, "extra_demonstrations").at(2)) simulator steps
to reveal its budget, while the recovery arm made
#count(row-for(budget-logged, "recovery_aggregation").at(1)) calls and discarded
#count(row-for(budget-logged, "recovery_aggregation").at(3)) recommendations to
reveal the same number of labels. That is the honest price of the method, and it is quantified in
@study-results rather than described qualitatively.

== Evaluation slices

Three slices are evaluated, all crossed over the identical ordered panel with
identical per-scenario schedules for every arm and bundle:

/ clean: no corruption. Answers whether the added labels helped nominal
  behavior at all.
/ matched: $g_+$ at the collection times. Answers whether the policy learned to
  handle the exact perturbation it was trained under. Findings here are
  perturbation-family-specific by construction.
/ unseen: $g_-$ at a disjoint time set. *Primary.* Answers whether the added
  labels bought anything beyond the operator and schedule that produced them.

== Eligibility, decided before any outcome

A scenario enters the unseen panel only if the nominal oracle path is longer
than the latest scheduled unseen corruption time. Otherwise the episode would
be over before the corruption could be delivered, and the scenario would
contribute a guaranteed non-event to every arm alike.

The filter uses only the fixed nominal oracle path length, a property of the
scenario computed before any policy existed, and retains #eligible-count of
#candidate-count test candidates. No learner output influences it. This is
eligibility, decided in advance for all arms at once, and it is emphatically
not an outcome-dependent exclusion.

== The decision rule, written down first

The interpretation of the interval was fixed at freeze time, before the panel
was opened:

#align(center)[
  #table(
    columns: 2,
    align: (left, left),
    table.header([*state*], [*condition on the 95% paired _t_ interval*]),
    [support], [lower bound above zero],
    [adverse], [upper bound below zero],
    [rule out], [interval contained within ±SESOI],
    [inconclusive], [anything else],
  )
]

Confirmatory status additionally requires at least $max(5, R_"train")$ complete
bundles. The precision target is reported with a met or unmet flag and does not
by itself change the status. Writing this down first is what makes the reported
state a finding rather than a choice made after seeing the numbers.
