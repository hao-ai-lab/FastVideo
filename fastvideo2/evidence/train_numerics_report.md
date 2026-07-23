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
