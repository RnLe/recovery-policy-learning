# Foundations lab02: Decision processes: from MDP to POMDP

**Status:** EXPLORATORY, FOUNDATIONS (pedagogical companion; not confirmatory evidence)

**Question answered:** What is a POMDP, why is this task one, and what would the alternatives cost?

Reproduce with `uv run grf run lab02 --force` (deterministic; named seeds under the gr-foundations domain).

## The formal object

A Markov decision process (MDP) is (S, A, T, R): states, actions, transitions, reward. A *partially observable* MDP adds an observation space and an observation function, because the agent never receives the state s, only an observation o = O(s). The mapping from each symbol to this repository's code is exported in `pomdp_mapping.typ`; here the dynamics are deterministic and all randomness sits in world generation (the reset seed).

## Aliasing, measured

Across 300 nominal oracle episodes (3928 visited states), the states collapse into 3746 distinct observations; 54 observation classes are *aliased* (the same bytes arise from provably different world states), 54 of them across entirely different mazes, and 5 carry *conflicting oracle actions*. Consequence: any memoryless policy, meaning any function from single observations to actions, must disagree with the oracle on at least 3.0% of visited states on this distribution. Memory is not a nicety; it is required for optimality.

## The showcase pair

`figures/aliasing_showcase.svg` shows two different mazes whose agents receive byte-identical observations while the oracle recommends different actions (`left` vs `forward`). Selection rule (disclosed): the conflicting cross-world class whose first member appears earliest in the rollout order, so no cherry-picking by appearance.

## Alternatives and why we reject them

Full observability (`figures/full_observability_contrast.svg`): MiniGrid can hand the policy the entire grid, turning the task into an MDP, but no physical agent observes the world state, and the study is about acting under realistic perception. Frame stacking approximates short memory with a fixed window; belief states are exact but require a known world model. The study's choice, learned memory in a recurrent network, is built and ablated in Labs 4 and 5.

## Bridge to the study

The study never constructs anything beyond this POMDP interface: policies consume exactly `StepResult` fields (image, direction, mission) plus their own previous executed action. The oracle, in contrast, *does* read the full state, and that asymmetry (privileged teacher, partially observing student) is what makes expert labels informative, and is the subject of Lab 3.
