= Background in three ideas <study-background>

Part II develops each of these from the ground up with its own measurements
(@lab-shift, @lab-decision, @lab-measurement). This chapter is the compressed
version for a reader who starts here.

== Imitation under covariate shift

Behavioral cloning treats control as supervised learning: minimize prediction
error against expert actions on states the expert visited. The trouble is that
the trained policy is then deployed on states *it* visits, and those are not
the same distribution. One wrong action moves the agent somewhere the
demonstrations covered thinly, its error rate there is higher and unmeasured,
and the next state is drawn from an even worse distribution. Errors compound
along a trajectory rather than averaging out across it, which is why a policy
with excellent per-step accuracy can still fail the task.

The remedy family that DAgger introduced is to move the labels rather than to
add more of them: query the expert in states the learner actually reaches, so
that supervision covers the distribution the policy will face. The open
accounting question, and the one this study asks, is whether relocating the
labels beats simply buying more of the ordinary kind at the same price.

Naming matters here. This study aggregates oracle labels in learner-visited
post-corruption states and fine-tunes on them. It does not implement DAgger's
policy mixing or its regret analysis, so it is called *DAgger-style
recovery-state aggregation* and not DAgger.

== Partial observability and recurrence

The agent sees a 7×7 egocentric symbolic view of a maze of nine rooms, so the
task is a partially observed Markov decision process. The optimal action
depends on history, for instance on which rooms have already been searched, and
not only on the current view. Part II measures this directly: distinct
underlying states produce byte-identical observations for which the oracle
recommends conflicting actions, so no memoryless policy can be optimal
(@lab-decision).

The policy therefore carries a recurrent state over the fused observation,
instruction, and action-history features, and supervision is applied to history
windows rather than to isolated frames. This is not an architectural
preference; it is forced by the decision problem.

== Intention-to-treat evaluation

Every scenario assigned a scheduled corruption stays in the denominator,
including episodes that ended before the corruption could be delivered. The
reasoning is the same as in clinical trials, and the bias it avoids is
specific and easy to fall into.

Delivery is not an independent event. A weaker policy ends its episodes sooner,
so a scheduled corruption is more often never reached, so its assignments are
more often undelivered. Conditioning the analysis on delivery would therefore
drop exactly the hardest episodes from the weakest arm and flatter it. Delivery
is a post-treatment variable, and Part II demonstrates the resulting bias on
simulations with known ground truth (@lab-measurement).

The cost of the honest choice is that the reported rates are slightly diluted
by episodes where nothing was done to the policy at all. That dilution is
reported rather than corrected: assigned and delivered counts appear side by
side in @study-results.
