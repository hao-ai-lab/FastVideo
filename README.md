# fastvideo2 — the v2.1 MVP (branch `will/v2.1`)

A from-scratch, deliberately small substrate for the FastVideo big bet:
**post-training → inference-optimized serving**, designed so that both kinds of
agents — the ones that *build* the framework and the ones that will *operate*
video models inside products — get inspectable contracts, a ground-truth
oracle, machine-readable verification, and an identity-chained runtime.

This branch is a clean slate: everything except `LICENSE` was removed, and the
MVP supports exactly one model, **Wan2.1-T2V-1.3B**, end to end.

## The four surfaces

| Surface | Where | What it guarantees |
|---|---|---|
| **Contracts** | `fastvideo2/card.py`, `pipeline.py`, `loop.py` | Cards are frozen *data* (no callables): JSON round-trip, stable content digest — the identity used by deploy configs, trainers, and RL environment manifests alike. Pipeline stage edges (`reads`/`writes`) are enforced at run time. Loop classes carry a `semantics` id and provenance pins it: distilled weights cannot silently run under a base sampler. |
| **Reference** | `fastvideo2/wan21/reference.py` | The complete model in one standalone eager file — the textbook an agent copies, and the oracle the production path is measured against. Never imported by production code. |
| **Verifier** | `fastvideo2/verify.py`, `fastvideo2/evidence/` | Tiered gates: T0 contracts (CPU, seconds) → T1 component fingerprints vs a blessed baseline → T2 trajectory parity vs the reference, with tolerance calibrated by measured run-to-run self-noise (the determinism contract) → T3 decoded-output parity + anti-degeneracy anchors. Every run appends typed records (card digest + env fingerprint) to the evidence ledger. |
| **Trace** | `fastvideo2/engine.py`, `loop.py` | Every unit of work is named `request/stage/loop.step`; the same identity chain lands in the returned trace (typed timings) and in nested NVTX ranges, so Nsight correlates kernels to model-level identity for free. |

## Quickstart

```bash
# machine-readable capability discovery
python -m fastvideo2 describe wan2.1-t2v-1.3b

# contracts only — CPU, no weights, no torch
python -m fastvideo2 verify wan2.1-t2v-1.3b --tier 0
pytest                                        # the same contracts, as tests

# GPU: bless the component baseline once, then gate against it
python -m fastvideo2 verify wan2.1-t2v-1.3b --tier 1 --bless
python -m fastvideo2 verify wan2.1-t2v-1.3b --tier 3

# generate
python -m fastvideo2 generate wan2.1-t2v-1.3b --prompt "a cat surfing a wave" --out cat.mp4

# the oracle, standalone (this file works copied out of the repo)
python -m fastvideo2.wan21.reference --prompt "a cat surfing a wave" --out ref.mp4
```

Weights resolve from the HF cache (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`) or an
explicit `--root`; components are stock diffusers/transformers modules, so
there is no conversion step and `load_component()` works standalone in a REPL.

## Design lineage (what this MVP encodes)

- **Cards as declared constants; variants as `derive()` diffs** — no builder
  functions, no factory bags, no toy backends welded into production cards.
- **The card digest is the axle artifact**: the same identity a deploy config
  points at, a trainer stamps provenance into, and an RL environment manifest
  pins (`substitution: exact | bounded | quality-changing` is already on
  `Provenance` for the post-training flywheel).
- **Typed conditioning** (`WanForwardInputs`): a new control channel is a new
  field the forward must consume — never a silently dropped kwarg.
- **Verification is the product**: gates fail closed, evidence is append-only
  data, baselines and tolerances are human-owned.
- **One loop, runtime-visible**: the driven-loop contract is what sessions,
  interleaved serving, and RL rollout branching will consume next; the engine
  stays a deliberately dumb one-shot runner until those consumers land.

## Scope and non-goals (MVP)

In: Wan2.1 T2V, bidirectional, single GPU, one-shot generation, tiers T0–T3.
Out (next, in order): causal/self-forcing students + sessions with forkable
state, the post-training flywheel emitting derived cards + evidence, the RL
environment server (`reset/step/branch`) over the same contracts, additional
model families via `derive()` and new recipe packages.
