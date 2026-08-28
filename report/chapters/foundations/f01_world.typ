#import "../../generated/foundations/lab01/world_facts.typ": *

= The world <lab-world>

*Intuition.* Before anything can be learned, there must be a world to act in.
Ours is deliberately austere: a small maze of connected rooms, a handful of
colored objects, an agent that can turn and walk, and a one-line instruction
such as "go to the grey box". Austerity is the point: every later claim about
learning, failure, and recovery can be checked by looking at the world
directly.

*Definition.* MiniGrid is a family of procedurally generated grid worlds;
BabyAI adds language missions and levels on top of it. The study's frozen task
is #raw(env-id): each reset seed generates a fresh maze on a #grid-shape cell
grid, places objects, and issues a mission. The episode succeeds when the
agent stands next to the requested object facing it, and ends unsuccessfully
after #max-steps steps (the environment's own limit).

*What the agent observes* is not the maze. The observation is a symbolic
$7 times 7 times 3$ integer tensor, an egocentric, occlusion-aware crop of
the cells in front of the agent, plus the view direction (0–3) and the
mission string. The three channels are lookup indices into fixed vocabularies
(object kind, color, door state), not pixels; walls block sight, so much of a
typical view is the "unseen" symbol.

#figure(
  image("../../generated/foundations/lab01/figures/observation_anatomy.svg", width: 100%),
  caption: [What the policy receives. Left: the world (render only, never
    observed). Right: the three integer planes of one observation; `?` marks
    cells occluded by walls. The view direction and mission string complete
    the input.],
)

*What the agent can do.* The study freezes three actions,
#frozen-actions, out of MiniGrid's seven. Turning rotates the view in place;
`forward` advances one cell when nothing blocks it. Why the other four actions
are excluded, and why the set must stay frozen, is derived when corruption
operators are introduced (@lab-shift).

#figure(
  image("../../generated/foundations/lab01/figures/action_effects.svg", width: 100%),
  caption: [The frozen action set, applied from one start state. The bright
    cone is the agent's field of view.],
)

*Measured census.* Over #census-seeds generated worlds, every mission follows
one template (`go to a/the <color> <kind>`), spanning #unique-missions
distinct strings across six colors and three object kinds, a tiny but real
language input: the mission decides which object counts as success. With the
contract's `doors_open: true` every door in the census is open; with the
environment default most are closed. The full tables (vocabularies, mission
distribution, door contrast) are generated in
`report/generated/foundations/lab01/`.

*Common misconception.* "The agent sees the maze." It never does: the world
exists in full only inside the simulator. Everything downstream, meaning the need for
memory, the value of a privileged teacher, and the meaning of recovery, follows
from the gap between the world's state and the agent's observation, which the
next chapter formalizes.

*Bridge to the study.* The study wraps exactly this environment in
`grounded_recovery.world.WorldSession`, adding contract checks rather than
features: resets require an explicit seed, only frozen actions pass, stepping
after termination is an error, and the reset world can be hashed into a
scenario identity used by the manifests.
