# Foundations lab01: The world: BabyAI/MiniGrid

**Status:** EXPLORATORY, FOUNDATIONS (pedagogical companion; not confirmatory evidence)

**Question answered:** What is BabyAI, and what does the agent actually see and do?

Reproduce with `uv run grf run lab01 --force` (deterministic; named seeds under the gr-foundations domain).

## The environment

`BabyAI-GoToObjMazeS4-v0` is a maze of connected rooms on a 10x10 cell grid (measured over 500 generated worlds). Every episode places the agent somewhere in the maze, scatters objects, and issues a natural-language mission. The episode ends in success when the agent stands next to the requested object, or after 144 steps (the environment's own limit).

## What the agent observes

Not the maze. The observation is a 7×7×3 integer tensor: an egocentric, occlusion-aware crop of the world in front of the agent, plus the view direction (0–3) and the mission string. The three channels are symbolic lookup indices (object kind, color, door state), not pixels; see `figures/observation_anatomy.svg` and the vocabulary tables. This partial view is what makes the task a POMDP (Lab 2).

## What the agent can do

The study freezes three actions: left, right, forward (ids 0/1/2 of MiniGrid's seven). Turning rotates the view in place; `forward` advances one cell if nothing blocks it, see `figures/action_effects.svg`. Why the other four actions are excluded, and why the set must stay frozen, is the subject of Lab 6.

## Mission grammar

All 500 sampled missions follow one template (`go to a/the <color> <kind>`; 0 unmatched), spanning 18 distinct strings, with colors blue, green, grey, purple, red, yellow and kinds ball, box, key. The language input is tiny but real: the policy must read it to know which object counts as success.

## Doors

With the study's contract setting `doors_open: true`, all 4528 doors seen across 500 worlds are open; with the environment default, 915 of 915 doors across 100 worlds are closed. The study's choice keeps the frozen 3-action set sufficient. The full story is told with the corruption operators in Lab 6.

## Bridge to the study

The study wraps exactly this environment in `grounded_recovery.world.WorldSession`, which adds contract checks: resets demand an explicit seed, only frozen actions pass, stepping after termination is an error, and the reset world can be hashed into a scenario identity. Those checks are bookkeeping, not learning; the world itself is what this lab measured.
