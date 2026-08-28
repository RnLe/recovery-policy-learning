#import "../../generated/foundations/lab04/learning_facts.typ": *

= Learning paradigms <lab-learning>

*Intuition.* Two ways to learn to act: try things and keep what pays
(reinforcement learning), or copy someone who already knows (imitation
learning). This study is often mistaken for the former. It is the latter, and
the distinction is worth earning with running code rather than asserting.

*Definition.* Reinforcement learning maximizes expected return through
interaction,
$ max_pi space EE [ sum_t R(s_t, a_t) ] , $
learning from reward alone. Imitation learning fits the policy to labelled
expert decisions,
$ min_pi space EE [ ell (pi (o_(1:t)), a^*_t) ] , $
supervised learning on (observation history, expert action) pairs. The study
is pure imitation: no gradient anywhere depends on reward; reward is read
only at evaluation time, to *measure* success.

*Real RL, where it belongs.* Tabular Q-learning, implemented from scratch,
on a tiny fully observable single-room task: the state (agent position and
direction, with #qlearning-states states ever encountered) is read directly from
the simulator, and the greedy policy reaches the goal in
#qlearning-best-steps steps after a few hundred episodes. RL genuinely works
when the state is visible and enumerable and reward is reachable.

#figure(
  image("../../generated/foundations/lab04/figures/qlearning_curve.svg", width: 100%),
  caption: [From-scratch tabular Q-learning on a fully observable toy MDP:
    greedy return (left) and steps to goal (right) during training.],
)

*Why that recipe does not carry over.* The study task generates a fresh world
per seed (no table can enumerate it; @lab-decision counted thousands of
distinct observations in a few hundred episodes), hides the state behind a
$7 times 7$ view, conditions success on a mission string, and pays reward
only at the end. None of this makes deep RL impossible: random walking
already succeeds 27.3% of the time (@lab-oracle), so exploration would find
reward. The study is not asking an RL question. It asks a
supervision-economics question (*which labels help more*, @lab-shift), and
behavior cloning is the controlled substrate for answering it.

*The first cloned policies.* #dataset-episodes oracle demonstrations
(#dataset-labels labelled steps) were cloned into two architectures on
identical data, identical optimizer, three seeds each: a memoryless policy
and the study's recurrent one (all training on #bc-device).

#figure(
  image("../../generated/foundations/lab04/figures/accuracy_vs_success.svg", width: 96%),
  caption: [The two imitation lessons. Bars: means over three seeds; whiskers:
    seed range.],
)

Two lessons, both visible in the figure. (1) *Accuracy is not success*: the
memoryless policy matches the oracle on #memoryless-open-acc of held-out
expert steps yet completes only #memoryless-unseen of unseen episodes
closed-loop, because small per-step errors compound over a rollout, the phenomenon
@lab-shift makes central. (2) *Memory matters*: the recurrent policy reaches
#recurrent-open-acc accuracy and #recurrent-unseen unseen success, the
end-to-end trace of @lab-decision's aliasing bound. (A caveat on effect
sizes at three seeds is developed in @lab-architecture.)

*Common misconception.* "High step accuracy means a good policy." The
memoryless column refutes it: closed-loop evaluation multiplies per-step
errors; open-loop accuracy quietly conditions on the expert's own states.
This is why the study's primary endpoint is closed-loop task success, never
prediction accuracy.

*Bridge to the study.* The study trains the same recurrent architecture with
the same masked cross-entropy idea, but on one-target-per-window items with
exact revealed-label budgets instead of whole episodes, a tightening whose
reason (@lab-measurement) is fairness accounting, not modelling.
