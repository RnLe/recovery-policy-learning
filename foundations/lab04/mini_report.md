# Foundations lab04: Learning paradigms: reinforcement versus imitation

**Status:** EXPLORATORY, FOUNDATIONS (pedagogical companion; not confirmatory evidence)

**Question answered:** Is this reinforcement learning, and if not, what is it?

Reproduce with `uv run grf run lab04 --force` (deterministic; named seeds under the gr-foundations domain).

## Two paradigms, one line each

Reinforcement learning maximizes expected return by interacting with the world and learning from reward. Imitation learning fits a policy to labelled expert decisions, which is supervised learning on (observation history, expert action) pairs. The study is pure imitation: no gradient anywhere depends on reward; reward is only read at evaluation time to *measure* success.

## Real RL, where it belongs

Tabular Q-learning, implemented from scratch, on `MiniGrid-Empty-5x5-v0` (fully observable, 32 states ever encountered): the greedy policy reaches the goal in 6 steps after a few hundred episodes (`figures/qlearning_curve.svg`). RL genuinely works when the state is visible and enumerable and reward is reachable.

## Why that recipe does not carry over

`rl_contrast` table: the study task generates a fresh world per seed (no table can enumerate it), hides the state behind a 7x7 egocentric view, conditions success on a mission string, and pays reward only at the end. None of this makes deep RL *impossible*: random walking already succeeds 27.3% of the time (Lab 3), so exploration would find reward. The study is not asking an RL question. It asks a supervision-economics question (which *labels* help more, Lab 6), and behavior cloning is the controlled substrate for answering it.

## The first cloned policies

200 oracle demonstrations (2663 labelled steps) were cloned into two architectures, with identical data, identical optimizer, three seeds each: a memoryless policy and the study's recurrent one. Training losses in `figures/bc_learning_curves.svg`; all runs on cuda.

## The two imitation lessons

(1) *Accuracy is not success*: the memoryless policy matches the oracle on 90.2% of held-out expert steps yet completes only 69.7% of unseen episodes closed-loop because small per-step errors compound over a whole rollout (Lab 6 makes this the central phenomenon). (2) *Memory matters*: the recurrent policy reaches 92.0% accuracy and 83.7% unseen success. Lab 2 proved a memoryless policy must disagree with the oracle on ≥3% of visited states; here the gap is visible end to end.

## Bridge to the study

The study trains the same recurrent architecture with the same masked cross-entropy idea, but on one-target-per-window items with exact revealed-label budgets instead of whole episodes, the tightening exists so that arms can be compared at equal label counts, which is Lab 7's subject. Evaluation there is exactly the greedy closed-loop rollout used here.
