= Reproduction <study-reproduction>

The complete pipeline is deterministic given the frozen contract. From a clean
checkout with `uv`, the pinned toolchain, and a system `ffmpeg` (needed only
for the rollout animations):

```
uv sync --frozen
uv run gr smoke          --config configs/pilot.yaml
uv run gr make-manifests --config configs/pilot.yaml
uv run gr preflight      --config configs/pilot.yaml
uv run gr freeze         --config configs/pilot.yaml
for b in B00 B01 B02 B03 B04 B05; do
  uv run gr run-bundle --contract configs/experiment_contract.yaml --bundle $b
done
uv run gr integrity      --contract configs/experiment_contract.yaml --phase preopen
uv run gr evaluate-final --contract configs/experiment_contract.yaml
uv run gr analyze        --contract configs/experiment_contract.yaml
uv run gr audit          --contract configs/experiment_contract.yaml
uv run grf study-extras
uv run gr integrity      --contract configs/experiment_contract.yaml --phase release
uv run gr publish-result --contract configs/experiment_contract.yaml
uv run gr export-typst   --contract configs/experiment_contract.yaml
typst compile report/main.typ build/recovery-policy-learning-report.pdf
```

Named seeds are derived with SHA-256 from the root seed, the bundle
identifier, and a component name drawn from a closed registry, so a typo
produces an error rather than a quietly different random stream. The frozen
contract, the code surface, the manifests, the eligible panel, and every
released artifact carry content digests, recorded in `configs/freeze_record.json`
and the artifact manifests. The release-phase integrity command re-derives the
published summary from the raw episode rows rather than trusting the stored
file.

Pinned versions: Python 3.11, `torch` 2.13.0, `minigrid` 3.1.0, `gymnasium`
1.3.0, with exact resolutions in `uv.lock`.

Part II runs separately as `uv run grf run all`, deterministically, under a
seed namespace disjoint from the study's.

== Provenance, stated exactly

The code digest recorded at freeze time identifies the source tree that
produced the confirmatory result. After the test opening, the following
additions were made, none of them scientific: the rollout-media generator
together with a rendering accessor on the environment session, which is
presentation plumbing that changes no contract semantics; the foundations
package of Part II with its journey media and site-staging commands; the
descriptive audits of @study-results and the exploratory panels of
@study-exploratory, which read the already-stored episode rows and write into
their own files; and the website, which consumes only the staged evidence
bundles.

The shipped tree therefore *does not* reproduce the freeze-time code digest,
and it is worth saying so rather than leaving a reader to discover it. What
verifies the result is not a match between the current source and a recorded
digest; it is that the published summary recomputes, to numerical tolerance,
from the immutable episode rows of the single opening, which is what
`gr integrity --phase release` checks and what any reader can rerun. No frozen
protocol value, manifest, checkpoint, or confirmatory result artifact was
modified at any point.
