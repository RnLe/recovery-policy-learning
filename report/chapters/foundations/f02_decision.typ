#import "../../generated/foundations/lab02/aliasing_facts.typ": *

= Decision processes <lab-decision>

*Intuition.* A decision process is the minimal mathematics of acting over
time: where you can be, what you can do, what happens next, and what counts
as success. The one distinction that shapes this entire study is whether the
agent gets to see where it actually is.

*Definition.* A *Markov decision process* (MDP) is a tuple $(S, A, T, R)$:
states, actions, a transition rule $T: S times A -> S$ (deterministic here),
and a reward $R$. A *partially observable* MDP adds an observation space
$Omega$ and an observation function $O: S -> Omega$; the agent never receives
the state $s$, only $o = O(s)$. Our task maps onto this tuple exactly: the
generated table `pomdp_mapping` pairs each symbol with the code entity that
implements it (the full grid and agent pose are the state; the $7 times 7$
crop, view direction, and mission are the observation; reward is sparse and
terminal; the horizon is the step limit). All randomness sits in world
generation: given the reset seed, dynamics are deterministic.

*Aliasing, measured.* Partial observability is not an abstract worry; it is
countable. Across #n-episodes expert episodes (#n-states visited states,
collected with the synchronized oracle of @lab-oracle), the states collapse
into #n-observation-classes distinct observations. Of these,
#n-aliased-classes observation classes are *aliased*: the same bytes arise
from provably different world states, #n-cross-world-classes of them from
entirely different mazes, and #n-heterogeneous-classes carry *conflicting
oracle actions*: the same input, two different correct outputs.

#figure(
  image("../../generated/foundations/lab02/figures/aliasing_showcase.svg", width: 100%),
  caption: [Perceptual aliasing, exhibited. Two different mazes under the
    same mission produce byte-identical observations while the oracle's
    optimal actions differ. Selection rule (disclosed): the conflicting
    cross-world class whose first member appears earliest in rollout order.],
)

*Consequence.* Any memoryless policy, meaning any function from single observations
to actions, must disagree with the oracle on at least
#memoryless-error-lower-bound-pct of visited states on this distribution, no
matter how it is trained. Memory is not an architectural taste; it is
required for optimality. This bound reappears twice: when the first cloned
policies are compared (@lab-learning) and when the architecture's recurrence
is ablated (@lab-architecture).

*Alternatives, and why they are rejected.* MiniGrid can hand the policy the
entire grid (`FullyObsWrapper`), turning the task into an MDP, but no
physical agent observes the world state, and the study is about acting under
realistic perception. Frame stacking approximates memory with a fixed window;
belief states are exact but require a known world model. The study's choice,
learned memory in a recurrent network, is built in @lab-architecture.

#figure(
  image("../../generated/foundations/lab02/figures/full_observability_contrast.svg", width: 100%),
  caption: [The rejected alternative: full observability would dissolve the
    problem this study is about.],
)

*Common misconception.* "Partial observability is a nuisance to engineer
around." It is the problem class itself: with the state visible, recovery
from a corrupted action would be a lookup; with a partial view, the policy
must carry history and *notice* that something went wrong.

*Bridge to the study.* The study's policies consume exactly the POMDP
interface of this chapter, meaning image, direction, mission, plus their own
previous executed action, and nothing else. The oracle, by contrast, reads
the full state. That asymmetry (privileged teacher, partially observing
student) is what makes expert labels informative, and is the subject of the
next chapter.
