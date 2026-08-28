#import "../../generated/foundations/lab03/policy_facts.typ": *

= Policies and oracles <lab-oracle>

*Intuition.* A policy is any rule that turns what the agent sees into what
the agent does. Before learning one, it pays to measure how far non-learned
rules get, and to meet the teacher whose answers the whole study spends as
currency.

*The policy spectrum.* Three policies on identical scenarios: a uniformly
random policy succeeds in #random-success of episodes, an honest surprise:
the maze is small and 144 steps is long, so blind wandering does stumble onto
the target, though hopelessly inefficiently. A hand-written right-hand wall
follower reaches #wall-success. The scripted oracle solves #oracle-success
with a mean path of #oracle-mean-steps steps. The gap between "sometimes, by
luck" and "always, directly" is what learning has to close.

#figure(
  image("../../generated/foundations/lab03/figures/policy_spectrum.svg", width: 100%),
  caption: [Left: success rates of three non-learned policies on identical
    scenarios. Right: how long expert solutions are.],
)

*What the oracle is.* `BabyAIBot` (shipped with MiniGrid) is a scripted
planner with privileged access: it reads the full grid, the true POMDP state
that @lab-decision showed the agent never observes, and replans from a
subgoal stack at every step. It is not learned and never runs inside the
policy. It exists to answer one question at any visited state: _what should
be done here?_ That answer is an *expert label*. A demonstration is a
sequence of such labels along the oracle's own path:

#figure(
  image("../../generated/foundations/lab03/figures/labelled_trajectory.svg", width: 100%),
  caption: [One expert demonstration as the learner consumes it: a sequence
    of (observation, oracle action) pairs, ending in success.],
)

*Recovery competence, and a falsification attempt.* Two separate questions,
measured on #sync-pairs identical scenarios with one forced off-proposal
action at a random early step. First, the fact the study rests on: an
honestly informed oracle *recovers*, with #sync-honest-success success after the
deviation, which is what makes it able to label states the learner reaches
instead of the expert (@lab-shift). Second, an attempt to break the bot's
bookkeeping three ways: lying about which action was executed
(#sync-lied-success), never informing it (#sync-never-success), and calling
its replanner twice per step (#sync-double-success). Nothing degrades: the
bot holds a live reference to the environment and replans from the true
world state, and pure navigation barely uses the executed-action bookkeeping
(it matters for pickup/drop/toggle subgoals, which the frozen action set
excludes).

#figure(
  image("../../generated/foundations/lab03/figures/synchronization_experiment.svg", width: 88%),
  caption: [The falsification attempt: recovery competence (left bar) and
    three misuse protocols. On this movement-only task, none degrades the
    oracle.],
)

The honest conclusion: on this task the study's synchronization contract
("exactly one replan per executed step, always told the executed action") is
not fragility protection, it is *accounting* protection. An oracle query is
the unit of supervision the study budgets and ledgers, so the contract is
what makes "$N$ labels" a well-defined, auditable quantity.

*Common misconception.* "The oracle helps the policy at run time." It never
does: at evaluation the policy is alone. Oracle answers reach the policy only
as training labels, under explicit budget accounting.

*Bridge to the study.* All of the study's collectors and evaluators drive
episodes through one shared loop (`run_synchronized_episode`) implementing
the honest protocol above, and the oracle's recovery competence had to pass a
preregistered 0.99 preflight gate before any learning experiment ran.
