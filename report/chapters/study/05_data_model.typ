#import "../../generated/foundations/lab05/architecture_facts.typ": total-parameters

= Data representation and policy model <study-data-model>

The architecture is dissected component by component, with a matched five-way
ablation, in @lab-architecture; the naive episode-level training that this
pipeline tightens is developed in @lab-learning.

== Records

Each episode stores per-step arrays, meaning the symbolic image, the direction,
the previous, proposed, recommended and executed actions, the reveal and
perturbation flags, and the termination state, alongside a JSON sidecar with
provenance digests and a content-addressed checksum over the array bytes. The
checksum is over the arrays, never over the container bytes, so re-saving the
same data cannot change its identity.

Collection appends to hash-chained ledgers, and an independent recount re-reads
every stored episode before training and refuses to proceed on any
disagreement. Keeping proposed, recommended, and executed as three distinct
fields, rather than collapsing them, is what makes it possible to verify after
the fact that a label was attached to the state the world was actually in.

== Training items

One training item is a contiguous history window ending at exactly one revealed
target, with the prefix capped at 32 steps. That definition is the reason
equal-exposure accounting is mechanical rather than approximate: one sampled
item is one target exposure, so counting items counts supervision.

The alternative, training on whole episodes, quietly breaks the comparison. A
nominal demonstration is label-dense along its whole length, while a recovery
window is a short burst of labels after a corruption, so an episode-level batch
delivers systematically more new labels per update to the extra-demonstration
arm. Part II leaves exactly that leak open on purpose and measures it
(@lab-shift); the study closes it with the one-target-per-window rule.

Two details prevent off-by-one errors from becoming silent label corruption.
The START token exists only at absolute episode start, so a window clipped by
the prefix cap begins with the true executed action that preceded it rather
than with a fake start. Padding exists only inside collation and is masked out
of the loss, so a padded batch and an unpadded one produce the same gradients.

== Model

Separate embeddings for the object, color, and state channels feed two 3×3
convolutions and a linear projection. A word-embedding GRU encodes the mission.
The previous *executed* action and the view direction are embedded. A linear
fusion and one policy GRU produce one logit per frozen action. The whole
network is #total-parameters parameters, which is small enough to audit by
hand and is asserted against the generated shape walkthrough in
@lab-architecture rather than quoted from memory.

The `forward` signature admits only these inputs. Coordinates, goal positions,
oracle state, and arm identity have no parameter to arrive through, and a
signature test enforces it. The boundary is the type signature itself rather
than a convention, which is the only version of that guarantee that cannot
erode.

The loss is masked cross-entropy at target positions. Training uses AdamW,
global-norm gradient clipping, and with-replacement sampling from named-seed
generators. All model computation runs on the contract's pinned device, and
run-to-run bit-determinism of the resulting model state is revalidated per
device rather than assumed to hold across hardware.
