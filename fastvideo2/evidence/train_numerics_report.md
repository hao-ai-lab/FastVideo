# Training parity — wan2.1 finetune vs fastvideo-main (legacy stack)

Gate `anchor.train-finetune-main`: PASS (2026-07-23, main @ c459a189).

Config (goldens manifest): crush-smol processed parquet, 1 GPU, batch 1,
grad_accum 1, num_latent_t 8, lr 5e-5, wd 1e-4, betas (0.9,0.999), clip 1.0,
uniform timestep sampling, target = noise - latents, dit fp32 masters + bf16
compute (FSDP2 world-1 mixed precision on main; explicit fp32-master/bf16-
compute chain in `fastvideo2/train/finetune.py`), cfg_rate 0, seed 42.

| row | result |
|---|---|
| noisy/pred/loss @ step0 | **bitwise 0.0** |
| loss steps 1–4 | 5.2e-5 … 2.6e-3, inside the measured band |
| params.w0 | bitwise |
| gnorm / params.w5 | informational (grad-noise carriers) |

Tolerance is MEASURED, not chosen: main-vs-main across three capture runs
drifts up to 3.66e-3 in per-step loss (flash-attention backward atomics);
the gate band is 1.5x that. Step-0 loss is exactly repeatable on main
(0.0 across runs) and we match it bitwise — i.e., the forward, input
construction, loss, and optimizer-chain math are exact; only the
nondeterministic backward separates runs, ours indistinguishable from
main's own reruns.

Root cause found by the gate: main's `train_one_step` runs
`_normalize_dit_input` (VAE scaling) BETWEEN batch fetch and noise prep —
capturing latents at fetch time silently records the wrong tensors (3.03
loss delta). Recorded post-normalization.

Reuse invariant held: the trained module is the SAME `WanModelFV` the
serving path loads (training-mode forward proved bitwise against main's
FSDP2-wrapped port via pred.step0).

Next gates (see fastvideo2/train/PORT_NOTES.md): dataloader order parity,
VSA training, DMD2 (legacy CFG convention), self-forcing (rollout via
WanCausalDMDLoop), attn-QAT/QAD, DiffusionNFT+VideoAlign.

## VSA training — added 2026-07-23

Gate `anchor.train-vsa-main`: PASS. Config: FastWan2.1 checkpoint (carries
deterministic `to_gate_compress` weights — the base-checkpoint variant
random-inits them and needs an init-RNG gate later), VIDEO_SPARSE_ATTN,
sparsity ramp 0.2→0.8 over 5 steps (rate 0.2, interval 1), otherwise the
finetune config.

step0 noisy/pred/loss **bitwise 0.0** — the vendored VSA kernel path
(tile/untile + `fastvideo_kernel.video_sparse_attn` + per-step ramped
metadata) is exact under training. Later steps sit WELL inside main's own
measured self-noise: the block-sparse backward at 0.8 sparsity drifts main
itself by 0.171 in per-step loss across reruns (vs flash's 3.7e-3); our
worst diff is 0.080 — about half of main's own run-to-run spread.
Band = 1.5x measured, from the goldens manifest.

## DMD2 distillation — added 2026-07-23

Gate `anchor.train-dmd2-main`: PASS. Legacy `distillation_pipeline.py`
convention (the shipped-FastWan authority): student/teacher/critic all from
the Wan2.1 base, simulate rollout, interval-2 student updates, teacher CFG in
the DMD2 parameterization (w=3.5), critic flow-match loss in bf16, separate
AdamWs, num_latent_t 4, 5 steps.

| row | result |
|---|---|
| x0 rollout (steps 0–1, pre-first-update) | **bitwise 0.0** |
| generator losses (steps 1,3) | 4.5e-6 / 7.6e-4 — in 3-run band |
| critic losses (all 5) | ≤2.9e-5 |
| params.w5 student/critic | 8.8e-5 / 4.3e-6 (info) |

Band = 1.5x main-vs-main across THREE capture runs (7.2e-4 max; main's own
gen.step3 spans 0.3169–0.3176 — ours 0.3178, at the distribution's edge).
x0 hashes are gated exact only until the first student optimizer update;
after it, backward noise makes bitwise impossible by definition and losses
carry the signal.

Replay machinery: main draws ALL DMD randomness from the global torch RNG —
the capture wraps `_generator_multi_step_simulation_forward`/`_dmd_forward`/
`faker_score_forward` and records draws in call order (target idx, rollout
noises, dmd/critic timesteps+noises); the anchor replays them through
`fastvideo2/train/dmd2.py::DMD2Step` (the sigma machinery is the inference
loop's `dmd_inference_table` — training reuses inference definitions).
Capture pins uncond embeds by encoding the canonical Wan negative prompt
through main's own text stack (the recipe normally gets them via the
validation path).

## VSA+DMD2 sparse distillation — added 2026-07-23

Gate `anchor.train-vsa_dmd2-main`: PASS. The FastWan recipe composition:
VSA-built student on FastWan weights (deterministic gate projections),
DENSE-built scorers on the base checkpoint — discovered from main's own
loader rejecting gate keys for scorer roles: teacher/critic score via plain
flash, NOT sparsity-0 VSA (the train loop's "dense copy" metadata is ignored
by the flash impl). Student rollout runs sparsity 0.8.

x0 rollout bitwise through the first student update; all gen/fake losses
within the measured band (main-vs-main drifts to 4.8e-2 at step3 under
sparse-student updates — an order noisier than dense DMD2's 7.2e-4);
param slices ~1e-4 (info).

Deferred (documented): the shipped recipe random-initializes student gate
projections from the BASE checkpoint — an init-RNG parity gate for later,
same caveat as the VSA finetune gate.

## Attn-QAT training — added 2026-07-23

Gate `anchor.train-qat-main`: PASS. Finetune config with
`FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN` (main's fake-quantized
SageAttention-style Triton kernel from fastvideo_kernel, head_dim 128,
warp-specialize disabled on Blackwell). Port: `layers/attn_qat.py` (vendored
wrapper, verbatim flag set, fail-closed) + `WanModelFVQAT`.

step0 noisy/pred/loss **bitwise 0.0** — the fake-quant forward is exact.
Later losses within the measured band: main's own quantized-backward reruns
drift to 8.3e-2 by step 4 (vs flash's 3.7e-3) — ours mostly below main's own
drift. Cross-attention stays dense (main's backend replaces self-attention
only).

## QAD (attn-QAT + DMD2) — added 2026-07-23

Gate `anchor.train-qad-main`: PASS on the first run, under the strict dense
band (no widening needed). Legacy loader-gated recipe: env ATTN_QAT_TRAIN
applies to the GENERATOR only (main's component loader restores flash for
teacher/critic), guidance 2.0, steps [1000,757,522]. Our side: WanModelFVQAT
student + dense WanModelFV scorers through the same DMD2Step.

x0 rollout (fake-quant attention) bitwise through the first student update;
gen losses ≤5.9e-4, critic ≤1.2e-4, param slices ~5e-6. The serving half of
QAD (fastwan-qad-fp8 card) was already bitwise — post-training -> serving is
now parity-gated END TO END for the QAD pipeline.

NOTE: a modular-stack QAD gate (role-local backend yaml, STANDARD CFG,
guidance 3.0) is deferred — the two stacks differ in CFG parameterization;
this gate pins the legacy convention.

## Self-forcing training — added 2026-07-23

Gate `anchor.train-self_forcing-main`: PASS — **121/121 rollout forwards
bitwise**, all five per-step rollout outputs (x0) hash-exact, critic losses
within band. The training rollout drives the SAME `WanModelFVCausal` +
KV/cross caches the SFWan serving path uses (`train/self_forcing.py::
sf_rollout` replaying recorded draws through a validating `DrawCursor`);
DMD/critic losses via DMD2Step on the SelfForcingFlowMatchScheduler table.

Root cause found (cost one cycle): main's SF phases run under
autocast(bf16), and the CAUSAL blocks use PLAIN LayerNorms — which autocast
promotes to fp32 compute (the dense model's explicit FP32 norms are
autocast-insensitive, which is why every earlier gate passed without it).
One missing autocast flipped all 121 forwards from DIFF to bitwise.

Gate scope (v1, documented): critic-side + rollout parity with no generator
update in-window (`dfake_gen_update_ratio` > steps). UPSTREAM FINDING: the
generator backward through the KV-cache rollout raises an autograd in-place
error unless gradient checkpointing is enabled (the recipe sets
`--enable_gradient_checkpointing_type full`), and checkpointed recompute
reads MUTATED caches — the recomputed attention windows differ from the
forward pass. A checkpointing-aware generator-step gate is deferred; the DMD
loss math itself is already parity-proven by the dmd2/qad gates. Config:
240x416 (resolution-agnostic math), 21 latent frames (pins the block-count
draw and avoids the >21-frame VAE re-encode branch).

## Serving identity — added 2026-07-23

Gate `anchor.serve-identity`: PASS. `fastvideo2.serve` (async-job REST +
WebSocket) over the engine: a served fastwan-qad request produced latents
BITWISE identical to the offline SDK for the same request/seed
(sha 41c4817d3d0f666b both sides), and the WebSocket delivered one LIVE
progress event per denoise step (3/3) via the engine's new `on_step` hook —
true per-step streaming, which main's servers list as a TODO.

Environment note (in code): this venv's FastAPI websocket dependency
injection closes connections without invoking endpoints; the stream route
is registered at the starlette level.
