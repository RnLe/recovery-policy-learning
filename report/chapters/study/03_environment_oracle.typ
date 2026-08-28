= Environment and oracle <study-environment>

The world, the teacher, and the operator family are each built from scratch and
measured in @lab-world, @lab-oracle, and @lab-shift. This chapter states what
the frozen contract fixes and why.

== Environment

`BabyAI-GoToObjMazeS4-v0`: a 3×3 maze of 4×4 rooms with one distractor,
templated missions of the form "go to a/the {color} {type}", a limit of 144
steps, and observations consisting of a 7×7×3 symbolic image plus a view
direction and the mission string.

The contract sets the environment's own `doors_open` parameter to true, and
this was a pilot discovery rather than a convenience. With closed doors the
scripted oracle must emit `toggle` in order to pass through them. That action
lies outside the frozen three-action set, so it would either enlarge the action
set, and with it the whole corruption operator family, or make the oracle
unsupportable at exactly the states where recovery matters most. With open
doors the oracle is movement-only, and that is verified over every collected
trajectory rather than assumed.

== The frozen action set, and why it is frozen

The action set is exactly ${sans("left"), sans("right"), sans("forward")}$.

A corruption operator has to change every action it touches, since an operator
with a fixed point would silently turn some corruptions into no-ops. An
operator that changes every element is a *derangement*. On a three-element set
exactly two total derangements exist, the two 3-cycles

$ g_+ = (1, 2, 0), quad g_- = (2, 0, 1), $

and each is the other's inverse. One of them, $g_+$, is the collection
operator. The other, $g_-$, paired with a disjoint set of scheduled times, is
the unseen operator.

This is a hard structural fact, not a design choice, and it is pinned by an
exhaustive enumeration test. It also sets a real limit on the study, which is
stated plainly rather than buried: with only two derangements available, "unseen"
cannot mean an independently sampled corruption family. It means the unique
other derangement, delivered at times the collection operator never used. That
limit is revisited in @study-discussion.

== The synchronized oracle

The MiniGrid `BabyAIBot` is wrapped so that `replan(last_executed)` is called
exactly once per active simulator step and always receives the action that was
actually executed, never the policy's proposal. Double calls, skipped calls,
calls after termination, and recommendations outside the frozen action set all
raise rather than proceed.

The reason for this severity is that the failure it prevents is silent. If the
bot is told a different action from the one the world took, it replans from a
state that does not exist, and every label derived from that point onward is
attached to the wrong state. Nothing crashes; the data simply becomes wrong.
Part II attempts to falsify the wrapper by lying to it, skipping calls, and
double-calling it, and finds that the bot's own replanning is robust enough to
recover anyway (@lab-oracle). The contract is therefore not fragility
protection. It is accounting protection: it guarantees that a recorded label
belongs to the state it is recorded against.

== Operator preflight

Before any learner ever ran, both operator families were tested in isolation:
600 episodes per family, one forced corruption each, with the scripted oracle
driving the recovery. The oracle recovered in all 1200 delivered corruptions,
against a frozen gate of 0.99.

This gate exists to separate two very different explanations of a failure. If
the learner fails to recover after a corruption, that is a fact about the
learner. If the *oracle* could not recover either, the corruption was not a
recoverable perturbation in the first place and the endpoint would be measuring
task impossibility rather than policy quality. The preflight rules that
explanation out in advance.
