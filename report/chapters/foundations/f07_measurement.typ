#import "../../generated/foundations/lab07/measurement_facts.typ": *

= Measuring honestly <lab-measurement>

*Intuition.* By this point the question is sharp: under equal additional
label budgets, do recovery labels beat extra demonstrations? What remains is
everything that makes an answer *trustworthy*, and every piece of that
discipline exists because a specific, demonstrable bias would otherwise
creep in. This chapter demonstrates the biases.

*Intention-to-treat, defined.* Analyze by what was *scheduled*, not by what
happened. Every perturbed evaluation episode has a scheduled corruption time;
sometimes the episode ends first and the corruption is never delivered.
Delivery depends on the policy's own behavior, so it is a post-treatment
variable, and conditioning on it compares filtered, non-comparable subsets.

*The bias, with known ground truth.* In a simulation built so that the true
ITT effect is #sim-true-itt by construction (and the better arm, being more
efficient, ends more episodes before late corruptions arrive), the ITT
estimator's bias is #sim-itt-bias while the per-protocol estimator, using only
delivered episodes, is off by #sim-pp-bias: a bias of the same order as the
effects this field measures, produced by nothing but a filter that looks
innocuous.

#figure(
  image("../../generated/foundations/lab07/figures/itt_bias.svg", width: 92%),
  caption: [2,000 simulated replicates against a known truth: the ITT
    estimator centers on the estimand; conditioning on delivery does not.],
)

On the real rows of @lab-shift the two estimates happen to agree
(#real-itt-delta ITT versus #real-pp-delta delivered-only) because delivery
is nearly universal there. The design still preregisters ITT: delivery
*could* differ between arms, and nothing in the data would warn you.

*Budget matching.* The study's currency is revealed oracle labels, and the
match was deliberately broken to see what happens: the recovery arm reran
with twice the budget and landed at #unmatched-unseen unseen success against
#matched-recovery-unseen matched. The feared inflation did *not* materialize
This base sits near its headroom ceiling (@lab-shift), so doubled
supervision bought nothing measurable in a single replicate. That is the
deeper point: an unmatched design attributes to the *method* whatever the
extra *budget* did or did not do, and a small unfrozen run cannot even tell
you which way the confound cuts. Matching is a design necessity for
attribution, not an empirical convenience.

#figure(
  image("../../generated/foundations/lab07/figures/unmatched_confound.svg", width: 84%),
  caption: [The unmatched design, tried on purpose. Whatever the outcome,
    the comparison stops measuring the method.],
)

*Why paired replicates.* Every pipeline bundle re-rolls collection and
training, shifting both arms together; analyzing per-bundle *differences*
cancels the shared noise. In simulation at six bundles, the paired analysis
detects a $+0.05$ gap with #paired-power power against #unpaired-power for
the unpaired analysis of the same data:

#figure(
  image("../../generated/foundations/lab07/figures/paired_power.svg", width: 88%),
  caption: [Interval widths over 2,000 simulated replicates: pairing turns
    six bundles into a usable instrument.],
)

*Freezing, and what it is made of.* A frozen protocol is a set of mechanical
commitments, not a promise. Two of them in miniature: flipping a single
contract field changes the contract's canonical fingerprint completely, so a
quiet edit cannot survive comparison against the recorded one; and editing one
row of a five-row hash-chained ledger is caught at exactly row
#chain-mismatch-row when the chain is recomputed. The full mechanism table:

#include "../../generated/foundations/lab07/freeze_mechanisms.typ"

*Common misconception.* "Preregistration is bureaucracy." Each mechanism
above is a bias with a name, blocked mechanically: selective delivery
(ITT), unequal generosity (budget match), shared-noise dilution (pairing),
silent redesign (contract hash), quiet data edits (chained ledgers),
rerun-until-pretty (single receipted opening), and post-hoc interpretation
(the preregistered claim decision rule).

*Bridge to the study.* Part II's endpoint is the eligible unseen
one-corruption ITT success difference, analyzed as a paired $t$ interval
across six bundles against a 0.05 smallest effect of interest, under a
preregistered claim decision rule, on data whose ledgers recount cleanly and
whose test set was opened exactly once against a receipt. Every one of those
words was one of this chapter's demonstrations, done, in Part I, at full
scale. You now have everything needed to audit it.
