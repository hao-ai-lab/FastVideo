# Training port — design + authority map (wan2.1 family)

Goal: the full post-training surface (finetune, VSA train, DMD2, VSA+DMD2,
attn-QAT/QAD, self-forcing, DiffusionNFT RL) on fastvideo2, each gated
against fastvideo-main by captured per-step loss goldens, same method as the
inference anchors.

## Invariants (additions to the repo-wide ones)

1. **Training reuses inference definitions.** The DiT is the SAME card-loaded
   module (`wan21/model_fv.py` classes); self-forcing rollout drives the SAME
   `WanCausalDMDLoop`/`LoopRunner`; DMD student rollout drives `WanDMDLoop`
   semantics. No training-side forks of forwards or samplers. (Causal cache
   writes already `.detach()`; nothing in the loops assumes no_grad.)
2. **Loss parity is the gate.** For each method: capture main's per-step
   losses (fixed seed, fixed data, deterministic algs, matched topology —
   1 GPU first) via a capture shim on dlrun; our trainer must reproduce them
   within the declared band (bitwise where kernels allow; flash/cuDNN
   backward nondeterminism gets a measured self-noise band, recorded in the
   manifest).
3. Data is pre-encoded parquet (latents + UMT5 embeds) — the trainer never
   touches the VAE/text encoder. Gate dataset:
   `wlsaidhi/crush-smol_processed_t2v`.

## Authority map (fastvideo-main, commit pinned per capture manifest)

- Finetune (legacy stack, authoritative for shipped models):
  `fastvideo/training/wan_training_pipeline.py` + `training_pipeline.py`.
  Loss `_transformer_forward_and_compute_loss:391`: flow-match MSE, target
  `noise - latents` (or precondition x0 via sigmas), float()ed, mean,
  / grad_accum. Inputs `_prepare_dit_inputs:278`: noise randn(CUDA gen
  seed+rank); u via compute_density_for_timestep_sampling (CPU gen
  seed+rank); `get_sigmas` (training_utils.py:97) EXACT `timesteps == t`
  table match; noisy = (1-s)*latents + s*noise; timestep fed to model as
  BF16. Scheduler FlowUniPCMultistepScheduler. AdamW, grad-clip, dit fp32 +
  bf16 autocast. Existing 2-GPU fixtures: train_loss 0.13823604 (thr 0.0025)
  and VSA 0.27654331 (thr 0.04) in fastvideo/tests/training/.
- VSA training: same pipeline + VIDEO_SPARSE_ATTN env; sparsity ramp
  training_pipeline.py:559 `min(step//interval, s//rate)*rate`; metadata
  `_build_attention_metadata:356`.
- DMD2 (legacy authoritative): `distillation_pipeline.py` — student/teacher
  (real_score)/critic (fake_score); DMD loss `_dmd_forward:591` with DMD2
  CFG parameterization `cond + w*(cond-uncond)`; grad
  `(fake-real)/|x0-real|.mean()`; `0.5*MSE(x0,(x0-grad).detach())`; critic
  flow-match `faker_score_forward:671`; student updated every
  generator_update_interval (5); separate critic AdamW; EMA_FSDP. NOTE: the
  new modular stack's dmd2.py uses STANDARD CFG (guidance offset by 1) —
  parity targets the LEGACY convention (shipped FastWan artifacts).
- VSA+DMD2: `scripts/distill/v1_distill_dmd_wan_VSA.sh`; VSA on student
  rollout only, teacher/critic scoring forced dense
  (distillation_pipeline train_one_step:816).
- Self-forcing: `self_forcing_distillation_pipeline.py` — rollout
  `_generator_multi_step_simulation_forward:141`: per-block random exit
  indices broadcast across ranks (`generate_and_sync_list:86`), no-grad
  non-exit steps, grad at exit + last-21-frame window, context rerun with
  store_kv; composes with inherited DMD losses; text-only data (latents
  rollout-generated); needs FASTVIDEO_FSDP2_AUTOWRAP=0; teacher is 14B.
- attn-QAT/QAD: ATTN_QAT_TRAIN backend (`attn_qat_train.py`, Triton
  fake-quant STE in fastvideo_kernel, head_dim 128 only, fail-closed);
  role-local backend override (student QAT, teacher/critic flash). QAD =
  stage-1 QAT finetune -> dcp_to_diffusers export -> stage-2 DMD2
  [1000,757,522] on `weizhou03/HD-Mixkit-Finetune-Wan`. Weight NVFP4 QAT
  separate: `nvfp4_qat_train_config.py` + `fp4linear.py` STE (SM100+).
- RL: `fastvideo/train/methods/rl/diffusion_nft.py` (merged) — student/old
  (EMA)/reference (KL) roles; group-relative advantages
  `(r-mean)/(std+1e-4)` grouped by prompt; NFT positive/negative prediction
  loss + KL; custom ODE/SDE sampler (`common/sampling.py`) — replace with
  our loop per invariant 1. VideoAlign reward (`KwaiVGI/VideoReward`,
  Qwen2-VL Bradley-Terry; MQ/VQ/TA scorers) lives ONLY on PR #1476 branch
  `maint/pr1476-runtime-compat` @ 518aeab0b — vendor from there; checkpoint
  via VIDEOALIGN_CHECKPOINT_PATH (local clone, not auto-downloaded).

## Port order and gates

1. substrate + finetune (gate: 5-step per-step losses vs main @ 1 GPU)
2. VSA train (same gate; sparsity ramp asserted)
3. DMD2 (generator + critic per-step losses; legacy CFG convention)
4. VSA+DMD2 (sparse-student/dense-scorer split asserted)
5. self-forcing (rollout reuses WanCausalDMDLoop; exit-index RNG replicated)
6. attn-QAT then QAD (kernel-gated; stage-1/stage-2 losses)
7. DiffusionNFT + VideoAlign (reward parity first, then loss parity)

Multi-GPU (sp/FSDP) lands after single-GPU parity per method, gated by the
same fixtures at the matching topology.
