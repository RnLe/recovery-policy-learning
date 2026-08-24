# Foundations lab06: When cloning breaks: shift, corruptions, recovery

**Status:** EXPLORATORY, FOUNDATIONS (pedagogical companion; not confirmatory evidence)

**Question answered:** Why does cloning break under its own mistakes, and what are recovery labels?

Reproduce with `uv run grf run lab06 --force` (deterministic; named seeds under the gr-foundations domain).

## The failure mode, measured

Behavior cloning fits the expert's state distribution; at deployment the policy visits its *own* states, and each error feeds the next (Ross & Bagnell's compounding-error argument). `figures/shift_anatomy.svg`: a single corrupted executed action drops the base policy from 83% (clean) to the shown per-t* success on delivered episodes, and the mean distance to the expert path grows step by step after the corruption while clean rollouts stay close.

## The corruption operators, and why the action set is frozen

A corruption operator must change every action it touches (otherwise some corruptions would be no-ops), i.e. it must be a *derangement* of the action set. For three actions exactly two derangements exist: the two 3-cycles, each the other's inverse (`derangements` table): one is used during collection, the other is held out as the unseen operator. Changing the action set would change this entire operator family, and enlarging it would break oracle support (with closed doors the bot emits `toggle`, outside the frozen set, the study's `doors_open` pilot discovery, Lab 1). That is why the action set is a frozen contract field, not a tuning knob.

## Two remedies, one budget

Both arms start from the *same* base weights (trained on 2101 labels) and receive exactly 400 additional oracle labels. The extra arm spends them on fresh nominal demonstrations; the recovery arm spends them on learner-visited states: the base policy rolls out, one action gets corrupted, and the oracle labels the next 8 states it actually reaches. Optimization is matched (600 updates, fixed 12+4 base/arm batch mix). One fairness leak is deliberately left open and *measured*: an extra-demo batch carries 54.2 new labels per update versus 29.7 for recovery episodes, because full demonstrations are label-dense while recovery windows are sparse. The study closes exactly this leak with one-target-per-window training items, as Lab 7 explains.

## What the mini-study shows

ITT success over 150 unseen scenarios x 3 replicates (`figures/three_arm_results.svg`): recovery reaches 84.7% on the unseen-operator slice versus 81.3% for extra demonstrations and 84.7% for the untouched base; per-replicate recovery-minus-extra deltas: +10.0%, +1.3%, -1.3%. Two honest observations. First, one corruption dents this base policy by only +2.0% on the ITT matched slice, and a strong base leaves little headroom for either remedy, one reason the study's frozen protocol checks perturbed competence and evaluates a far larger panel. Second, three replicates at this scale cannot even fix the *sign* of the difference, the small-n lesson Lab 7 turns into design requirements. The confirmatory answer, under the frozen protocol, is the study itself (+14.3pp, 95% interval +10.0 to +18.6pp, over 6 pipeline bundles and 536 eligible scenarios).

## Bridge to the study

Everything here is a scaled-down mirror of `grounded_recovery.experiment`: the study adds bit-exact arm cloning from a shared checkpoint, hash-chained exposure ledgers with a fairness audit, exact window-level target accounting, preregistered corruption time sets, and the one-opening evaluation discipline, the honest-measurement machinery that Lab 7 walks through.
