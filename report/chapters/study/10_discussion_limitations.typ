#import "../../generated/metadata.typ": *
#import "../../generated/primary_result.typ": *
#import "../../generated/audit_result.typ": *
#import "../../components/format.typ": *

= Discussion and limitations <study-discussion>

== What the comparison isolates

The design isolates label *allocation*. Both full-budget arms start from
bit-identical checkpoints, reveal exactly the same number of oracle targets,
and receive exactly the same optimizer updates on the same mix of old and new
targets. What differs is only *where* the labels came from: states on fresh
expert trajectories, or states the learner itself reached after a corrupted
action. A difference on the unseen slice is therefore attributable to
allocation, conditional on this environment, oracle, budget, and schedule.

That conditional clause is doing real work, and the rest of this chapter is
about it.

== The clean-slice result changes the reading

The recovery arm leads by
#signed-pp(row-for2(secondary-contrasts, "recovery_aggregation",
  "extra_demonstrations").at(3)) pp on the *clean* slice, where no corruption is
delivered at all, with an interval excluding zero.

The obvious story for this study would be that the recovery arm learned to undo
a particular corruption. The clean-slice result argues against that story. If
the advantage were operator-specific, it should vanish when no operator is
applied. Instead it persists, which points at the labels themselves: recovery
labels are collected in states the policy actually visits, including the
ordinary states it passes through in the eight steps after a corruption, so the
supervision covers the deployment distribution better regardless of whether a
corruption occurs.

That is the covariate-shift argument in its general form rather than a
perturbation-specific repair, and it is a stronger reading of the result than
the one the study set out to test. It is also, being a secondary contrast, not
prespecified, and it should be treated as a well-supported observation rather
than a confirmed claim.

== Trade-offs, stated beside the benefit

/ Acquisition cost: revealing #budget-b recovery labels took
  #count(row-for(budget-logged, "recovery_aggregation").at(1)) oracle calls
  against #count(row-for(budget-logged, "extra_demonstrations").at(1)) for the
  same number of demonstration labels, discarding
  #count(row-for(budget-logged, "recovery_aggregation").at(3)) recommendations
  along the way. The
  budgets that were matched are label budgets, not teacher-time budgets. With a
  human teacher the ranking could plausibly reverse, and nothing here tests that.
/ Path cost: among successful episodes the recovery arm's paths are modestly
  longer relative to the oracle than either comparison arm's
  (@study-results). The gain is in whether the task is solved, at a small cost
  in how directly.
/ Failed rollouts are still consumed: recovery collection runs the learner, and
  episodes that fail or truncate are stored rather than discarded, but they
  still consumed simulator time that the extra-demonstration arm did not spend.

== Limitations

+ *The evidence is in silico, symbolic, discrete, and oracle-supervised.*
  Nothing here demonstrates physical-robot, continuous-control, or human-teacher
  competence. The teacher is a scripted planner with privileged access to the
  full grid, which is exactly the kind of teacher real robotics does not have.

+ *"Unseen" is narrower than the word suggests.* On a three-action set exactly
  two total derangements exist, so the held-out operator is necessarily the
  inverse of the collection operator (@study-environment). Unseen-ness rests on
  the distinct operator together with a disjoint set of scheduled times, not on
  an independently sampled corruption family. This is a structural consequence
  of the action set, disclosed rather than sampled away, and it is the single
  largest qualification on the headline.

+ *Every failure is a timeout.* The environment contains no irreversible
  action, so the study cannot say anything about recovery from a mistake that
  cannot be undone (@study-failures). That is precisely the case where
  recovery labels would matter most in a physical system.

+ *The base policy was more competent than planned.* Perturbed competence sat
  near #pct(row-for2(mean-success, "bc_base", "unseen").at(2)) rather than in
  the planned 20 to 70 percent range, because a single corrupted action is
  cheap to recover from in an open maze. The non-saturation criterion of at
  least ten points of headroom still held, and the deviation is recorded in the
  pilot report rather than smoothed over.

+ *The replicate count is small, by design.* The paired _t_ interval is
  conditional on the frozen scenario panel and rests on #bundles-completed
  pipeline replicates, which is
  compute-bound rather than principled. Every per-bundle point is shown so that
  the reader can see the dispersion rather than infer it, and the
  extra-demonstration arm's own contrast against the base is not resolved at
  this count (@study-results).

+ *One environment, one task family.* All evidence comes from a single BabyAI
  configuration with templated "go to" missions. Nothing here establishes that
  the ranking holds for longer-horizon or compositional tasks.

#scope-note[
  *What this study does not establish.* That recovery-state labels beat extra
  demonstrations in general; that the result transfers to continuous control,
  real perception, or human teachers; that the comparison holds when teacher
  time rather than label count is the budget; or that any of this bears on
  reinforcement learning, which appears nowhere in the method.
]
