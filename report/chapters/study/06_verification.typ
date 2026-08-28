= Verification and reproducibility <study-verification>

Each mechanism in this chapter is demonstrated in miniature, on cases with
known ground truth, in @lab-measurement.

The test suite is an executable statement of the scientific contract rather
than a safety net for refactoring. The invariants below are chosen for what
their failure would do to the evidence, not for how hard they were to write.

/ Oracle synchronization: exactly one `replan` per active step, fed the action
  that was executed. Failure would attach labels to states the world was never
  in, and nothing would crash.
/ Exact budgets: base data holds exactly $N_0$ targets and each arm exactly
  $B$, with deterministic handling of a partial final window. Failure would
  break the budget-matched comparison that the whole study rests on.
/ Clone equality and fairness: bit-identical arm starts; equal updates, target
  draws, and cumulative exposures at every round boundary; an unchanged base
  digest after arm training. Failure would let an advantage come from
  optimization rather than from allocation.
/ Causality: logits at step $t$ are unaffected by inputs after $t$, padding is
  inert, and stepwise inference reproduces the training-time forward pass
  exactly. Failure would mean evaluating a different function from the one
  trained.
/ Determinism: the complete pipeline, from collection through training to
  checkpoints, reproduces bit-identical model state digests under the seed
  bundle.
/ Lifecycle: freezing refuses unresolved placeholders and double freezes; final
  commands accept only the frozen contract; exactly one receipted confirmatory
  opening exists per contract; and the release-phase integrity check recomputes
  the published summary from the raw episode rows rather than trusting it.

Scenario manifests are hashed over the reset world content rather than over the
seed, so two different seeds that happen to generate the same world cannot land
in two different splits and leak. All eight purpose splits are pairwise
disjoint by seed and by world content. Evaluation code never constructs the
oracle, so no oracle recommendation can reach a policy at test time.

Six of those eight splits carry the experiment: base data, recovery collection,
validation, operator preflight, the test candidate panel, and visualization.
The remaining two, `difficulty_shift` and `expert_diagnostic`, were reserved at
freeze time for optional descriptive diagnostics and were not part of the
confirmatory opening. They were opened afterwards, and what they show is in
@study-exploratory, labelled exploratory throughout. No split is generated and
then left unaccounted for.

The one-opening rule deserves its own sentence. The confirmatory evaluator
writes an append-only receipt, containing the panel identity and every
checkpoint digest, *before* it reads a single outcome. An interrupted run may
resume the same opening only from byte-identical inputs. There is no
discretionary reopening and no selective rerun of an inconvenient cell. This is
a process guard rather than a security claim: it does not make tampering
impossible, it makes tampering something that has to be done deliberately and
leaves a record.
