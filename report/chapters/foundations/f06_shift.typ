#import "../../generated/metadata.typ": bundles-completed, eligible-count
#import "../../generated/primary_result.typ": mean-delta, interval-lower, interval-upper
#import "../../components/format.typ": signed-pp, interval-pp
#import "../../generated/foundations/lab06/shift_facts.typ": *

= When cloning breaks <lab-shift>

*Intuition.* Behavior cloning is trained on the expert's states but deployed
on its own. The first wrong action moves the policy somewhere the
demonstrations never covered; whatever it does there is unsupervised, and
errors feed on themselves. This chapter measures that spiral, formalizes the
corruption operators used to trigger it on demand, and previews the study's
remedy comparison at small scale.

*The failure mode, measured.* The compounding-error argument (Ross and
Bagnell's classical analysis: per-step error $epsilon$ can cost on the order
of $epsilon T^2$ over a horizon $T$, precisely because mistakes change the
state distribution) becomes concrete with one corrupted executed action:

#figure(
  image("../../generated/foundations/lab06/figures/shift_anatomy.svg", width: 100%),
  caption: [Left: success of the base policy when exactly one executed action
    is corrupted at time $t^*$ (delivered episodes), against its clean rate.
    Right: mean distance to the expert path after the (aligned) corruption
    time, since clean rollouts drift apart from the expert's route slowly (valid
    alternative paths diverge positionally too); corrupted rollouts drift
    roughly twice as far.],
)

*Corruption operators, and why the action set is frozen.* A corruption must
change every action it touches, otherwise some corruptions would be no-ops;
an operator is therefore a *derangement* (fixed-point-free permutation) of
the action set. For three actions exactly two derangements exist: the two
3-cycles, each the other's inverse (`derangements` table): the study uses one
during collection and holds out the other as the unseen operator. Changing
the action set would change this entire operator family; enlarging it would
also break oracle support, since with closed doors the bot emits `toggle`, outside
the frozen set, which is exactly how the study's `doors_open: true`
parameterization was discovered during piloting (@lab-world). The action set
is a frozen contract field, not a tuning knob.

#include "../../generated/foundations/lab06/derangements.typ"

*Two remedies, one budget.* Both arms start from the *same* base weights
(trained on #base-labels labels) and receive exactly #budget-labels
additional oracle labels each. The *extra* arm spends them on fresh nominal
demonstrations. The *recovery* arm spends them where the learner actually
goes: the base policy rolls out, one action is corrupted, and the oracle,
kept synchronized along a trajectory it did not choose, labels the next
#recovery-window states the policy visits. Optimization is matched (same
update count, same fixed base/new batch mix). One fairness leak is
deliberately left open and *measured*: an extra-demo batch carries
#exposure-extra new labels per update against #exposure-recovery for
recovery batches, because demonstrations are label-dense while recovery
windows are sparse. The study closes exactly this leak with
one-target-per-window training items (@lab-measurement).

*What the mini-study shows.*

#figure(
  image("../../generated/foundations/lab06/figures/three_arm_results.svg", width: 100%),
  caption: [Intention-to-treat success by evaluation slice (three replicates
    as dots). Exploratory preview at small scale, not confirmatory
    evidence.],
)

Across three replicates the recovery-minus-extra difference on the
unseen-operator slice spans #delta-unseen-min to #delta-unseen-max (mean
#delta-unseen-mean), with corruption delivery at #delivered-rate. Two honest
observations. First, one corruption dents this base policy by only
#base-corruption-dent on the matched ITT slice, and a strong base leaves little
headroom for either remedy, one reason the study's frozen protocol checks
perturbed competence and evaluates a far larger panel. Second, three
replicates cannot even fix the *sign* of the difference, the small-$n$
lesson the next chapter turns into design requirements. The confirmatory
answer, over #bundles-completed replicates and #eligible-count eligible
scenarios under the frozen protocol, is Part I's
#signed-pp(mean-delta, digits: 2) pp with a 95% interval of
#interval-pp(interval-lower, interval-upper, digits: 2) pp (@study-results).

*Common misconception.* "Corruptions are adversarial attacks." They are a
*probe*: a controlled, disclosed way to place the policy off its training
distribution exactly once, so that recovery behavior becomes measurable and
comparable. The unseen operator exists so that the measurement cannot reward
memorizing the probe itself.

*Bridge to the study.* Everything here mirrors
`grounded_recovery.experiment` at reduced scale; the study adds bit-exact arm
cloning, hash-chained exposure ledgers with a fairness audit, exact
window-level target accounting, preregistered corruption time sets, and the
one-opening evaluation discipline of the next chapter.
