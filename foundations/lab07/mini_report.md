# Foundations lab07: Measuring honestly: budgets, ITT, and frozen protocols

**Status:** EXPLORATORY, FOUNDATIONS (pedagogical companion; not confirmatory evidence)

**Question answered:** What is intention-to-treat, and why budgets, pairing, and freezing?

Reproduce with `uv run grf run lab07 --force` (deterministic; named seeds under the gr-foundations domain).

## Intention-to-treat, defined

Analyze by what was *scheduled*, not by what happened. In the study every evaluation episode has a scheduled corruption time; sometimes the episode ends first and the corruption is never delivered. Delivery depends on the policy's own behavior, so it is a post-treatment variable, and conditioning on it compares filtered, non-comparable subsets.

## The bias, with known ground truth

A simulation (536 scenarios, 2000 replicates) where the true ITT effect is +0.070 by construction: the ITT estimator's bias is -0.0010; the per-protocol estimator (delivered episodes only) is off by -0.0355, a bias of the same order as the effects being measured (`figures/itt_bias.svg`). On Lab 6's real rows the two estimates agree (+3.3% ITT vs +3.8% delivered-only) because delivery is nearly universal there. The design still preregisters ITT, because delivery *could* differ between arms and nothing in the data would warn you.

## Budget matching

The study's currency is revealed oracle labels. We deliberately broke the match: the recovery arm reran with 800 labels (twice the budget), landing at 83.3% unseen success against 84.7% matched (`figures/unmatched_confound.svg`). The feared inflation did *not* materialize here, because this base policy sits near its headroom ceiling (Lab 6), so even doubled supervision buys nothing measurable in a single replicate. That is the deeper point: an unmatched design attributes to the *method* whatever the extra *budget* did or did not do, and a small unfrozen run cannot even tell you which way the confound cuts. Budget matching is a design necessity for attribution, not an empirical convenience.

## Why paired replicates

Every pipeline bundle re-rolls collection and training, shifting both arms together; analyzing per-bundle *differences* cancels that shared noise. In simulation at 6 bundles the paired analysis detects a +0.05 gap with 100% power versus 82% unpaired (`figures/paired_power.svg`). This is why the study runs six bundles and reports the paired-t interval, with a cluster bootstrap as sensitivity.

## Freezing, and what it is made of

A frozen protocol is a set of mechanical commitments, not a promise (`freeze_mechanisms` table). Two of them, in miniature: flipping a single contract field (`data.h`) changes the canonical hash from `e777efe2d8f83c39…` to `fe28e0752e1a6a13…`, so no quiet edit survives; and editing one row of a five-row hash-chained ledger is caught at exactly row 2 when the chain is recomputed.

## Bridge to the study

The study's endpoint is the eligible unseen one-corruption ITT success difference, analyzed as a paired t interval across six bundles against a 0.05 smallest-effect-of-interest, under a preregistered claim decision rule, on data whose ledgers recount cleanly and whose test set was opened exactly once against a receipt. Every one of those words is one of this lab's demonstrations, done at full scale.
