#import "../../generated/metadata.typ": *

= Question and scope <study-question>

== The question in words

Suppose a policy has already been cloned from #n0 expert-labelled decisions and
performs reasonably well. Someone offers you #budget-b more oracle labels. You
can spend them in two ways. You can ask the expert to demonstrate more tasks
from scratch, which produces more of exactly the kind of data you already have.
Or you can let your own policy act, corrupt one of its actions so that it ends
up somewhere it would not normally be, and ask the expert what to do from
there.

The second option is the DAgger family's answer to closed-loop distribution
shift, and its motivation is well known. What is far less often measured is
whether it actually wins when the two options are given the same budget, the
same starting weights, the same number of gradient steps, and the same number
of supervised targets per step. Without that matching, an apparent advantage
can come from having seen more data, or trained longer, or started from a
better checkpoint.

This study measures the comparison with all of those held fixed.

== The estimand

For method $m$, pipeline bundle $r$, and eligible unseen scenario $e$, let
$Y[m, r, e] in {0, 1}$ be intention-to-treat task success. With per-policy
rates $hat(p)[m, r] = "mean"_e Y[m, r, e]$, the primary contrast is

$ delta_r = hat(p)["recovery", r] - hat(p)["extra demo", r], quad
  hat(Delta) = "mean"_r delta_r , $

estimated with a prespecified 95% paired _t_ interval across complete pipeline
bundles, conditional on the frozen scenario panel. The smallest effect size of
interest is #sesoi absolute success.

Three choices inside that definition carry most of the weight, and each is
defended in its own chapter.

The *unit* is the pipeline bundle, not the episode. A bundle is one complete
run from base data through base training, arm cloning, four rounds of
collection and fine-tuning, to two final policies. Episodes inside a bundle are
repeated measurements of the same trained policies, so treating them as
independent replicates would shrink the interval by roughly the square root of
the panel size and claim a precision the design does not have (@lab-measurement).

The *comparison* is recovery against extra demonstrations. The untouched base
policy is reported throughout, but it is a context line, not the competitor.
Beating a policy that received no additional labels at all would say nothing
about how to spend a budget.

The *scoring* is intention to treat. A scenario assigned a corruption stays in
the denominator whether or not the corruption was delivered, and whether or not
the episode ended first (@study-design).

== In scope

A symbolic, discrete, fully scripted-oracle BabyAI study of label allocation
under closed-loop distribution shift, with exact budget and exposure
accounting, leakage-resistant scenario splits, and one receipted confirmatory
test opening.

== Out of scope

Reinforcement learning, in the strict sense that no reward term exists anywhere
in the update; physical robots; continuous control; human supervision;
pretrained perception or language models; and any claim that BabyAI evidence
transfers as such to real robotics. The precise name for the treatment is
*DAgger-style recovery-state aggregation*: adaptive data collection followed by
masked behavioral cloning. It is neither the full DAgger algorithm nor
reinforcement learning, and it is described that way throughout.

What the study can support, and what it deliberately cannot, is set out in
@study-discussion rather than left to the reader to infer.
