# Foundations lab03: Policies, oracles, and where labels come from

**Status:** EXPLORATORY, FOUNDATIONS (pedagogical companion; not confirmatory evidence)

**Question answered:** What is a policy, what is the oracle, and what exactly is an expert label?

Reproduce with `uv run grf run lab03 --force` (deterministic; named seeds under the gr-foundations domain).

## The policy spectrum

On 300 identical scenarios: a uniformly random policy succeeds in 27.3% of episodes, a genuinely measured surprise worth being honest about: the maze is small and 144 steps is a long time, so blind wandering does stumble onto the target occasionally (in 120 steps on average, versus 12.1 for the oracle). A hand-written right-hand wall follower reaches 54.3%, and the scripted oracle 100.0%. The reward signal is terminal-only, so what makes learning from reward hard here is not that success is unreachable but that a single end-of-episode bit must be attributed across up to 144 decisions, the credit-assignment framing Lab 4 makes precise. See `figures/policy_spectrum.svg`.

## What the oracle is

`BabyAIBot` (shipped with MiniGrid) is a scripted planner with privileged access: it reads the full grid, the true POMDP state Lab 2 showed the agent never observes, and maintains a subgoal stack it replans from at every step. It is not learned and not part of the policy; it exists to answer one question at any visited state: *what should be done here?* That answer is an expert label.

## Recovery competence, and a falsification attempt

Two separate questions were measured on identical scenarios (163 episodes each, one forced off-proposal action at a random early step). First, the fact the whole study rests on: an honestly informed oracle *recovers* 100.0% success after the deviation, which is what makes it able to label learner-visited off-path states (Lab 6). Second, we tried to break the bot's bookkeeping three ways: lying about the executed action (100.0%), never informing it (100.0%), and calling `replan` twice per step (100.0%). None of it degrades this task: `BabyAIBot` holds a live reference to the environment and replans from the *true* world state, and pure navigation barely uses the `action_taken` bookkeeping (it matters for pickup/drop/toggle subgoals, which the frozen action set excludes). The honest conclusion: on this task the synchronization contract is not fragility protection, it is *accounting* protection. An oracle query is the unit of supervision the study budgets and ledgers, so "exactly one replan per executed step" is what makes "N labels" a well-defined, auditable quantity.

## Where labels come from

`figures/labelled_trajectory.svg` shows one expert episode as the learner will consume it: a sequence of (observation, oracle action) pairs. A *demonstration* is nothing more than this sequence collected along the oracle's own path; a *recovery label* (Lab 6) is the same query issued at a state the learner reached instead.

## Bridge to the study

The study's collectors and evaluators all drive episodes through one shared loop (`run_synchronized_episode`) that threads the executed action back into the oracle, the honest protocol above. Labels only ever enter datasets through explicit budget accounting (`revealed_targets`), which is what makes the later arm comparison fair (Lab 7).
