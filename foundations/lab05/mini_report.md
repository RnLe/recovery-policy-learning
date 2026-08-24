# Foundations lab05: The policy network, piece by piece

**Status:** EXPLORATORY, FOUNDATIONS (pedagogical companion; not confirmatory evidence)

**Question answered:** What exactly is the policy network, and does every piece earn its place?

Reproduce with `uv run grf run lab05 --force` (deterministic; named seeds under the gr-foundations domain).

## The model at a glance

179,507 parameters: three channel embeddings over the symbolic 7x7 view, two 3x3 convolutions, a linear projection; a word-embedding + GRU mission encoder; direction and previous-action embeddings; one fusion layer; a GRU policy core; a three-logit head. The full table in `tables/shape_walkthrough.csv` is *generated from a live instance*, so it cannot drift from the code. A from-scratch reimplementation (`gr_foundations.models.LabPolicy`) matches the study model parameter for parameter (asserted at runtime and in tests).

## One component removed at a time

Five variants, identical demonstrations, identical optimizer, 3 seeds each (`figures/ablation_results.svg`): full 83.0%, no policy GRU 80.0%, no mission input 83.3%, no previous-action input 83.3%, bag-of-words mission 80.3% closed-loop success on unseen scenarios.

## Reading the ablation honestly

The candid finding: at this training scale, clean-condition success is a blunt instrument. All five variants land within a few points (means 80.0%–83.0%), with heavily overlapping three-seed ranges (full 80%–85%, no-GRU 79%–82%). Lab 4 trained this same memoryless architecture on three other seeds and measured 69.7%; pooling all six seeds spans 64%–82%. Three-seed comparisons wobble, which is precisely why the study runs six paired replicates and reports an interval (Lab 7). Two results deserve their own sentences. First, *removing the mission costs nothing here* (83.3%, and mean episode lengths do not separate either): with one distractor and a 144-step limit, a mission-blind policy that simply tours objects still ends on the right one, because the endpoint tolerates detours, so language earns its keep only at tighter horizons or richer scenes (an interpretation, marked as such). Second, the bag-of-words mission encoder (80.3%) matches the GRU at this five-word grammar.

## Why the architecture is still the right one

The components are justified by their *roles in the study*, not by clean-run ablation wins. Memory: Lab 2 proved no memoryless policy can match the oracle on aliased states, a structural argument that holds regardless of seed noise, and Lab 4's own seeds showed the end-to-end gap. Previous-action input: nearly free in clean conditions, but it is the only channel through which an externally corrupted execution becomes visible to the policy; its purpose only exists under Lab 6's corruptions. Mission conditioning: it defines the task; that ITT success barely punishes its removal at this scale is a fact about the endpoint's tolerance, not about the input being uninformative. The study inherits the standard BabyAI treatment and makes no architectural novelty claims.

## Why a GRU and not a Transformer

The dataset is a few thousand labelled steps; the model runs closed-loop, one observation at a time, carrying state forward. A recurrent core consumes O(1) memory per step at inference, trains stably at this scale, and keeps the architecture small enough to audit by hand, which this repository treats as a feature, not a limitation.

## Bridge to the study

`grounded_recovery.model.RecoveryPolicy` is this exact architecture; the study adds nothing at model level. Its checkpoints additionally freeze the vocabulary, action names, and RNG states so that training can be resumed or cloned bit-exactly. The cloning matters in Lab 6, where three arms must start from the *same* base checkpoint.
