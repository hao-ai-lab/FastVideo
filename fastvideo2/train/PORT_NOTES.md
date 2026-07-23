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

## DMD2 build facts (read from distillation_pipeline.py, for #19)

- train_one_step (:807): collect grad_accum batches (fetch->normalize->
  prepare->attn metadata; metadata_vsa deepcopy, dense copy gets
  VSA_sparsity=0). Student phase only when
  `current_trainstep % generator_update_interval == 0`: rollout -> _dmd_forward
  -> backward -> clip(student) -> student AdamW -> EMA update. Critic phase
  EVERY step: faker_score_forward -> backward -> clip(critic) -> critic AdamW
  (+ its LR scheduler).
- Rollout `_generator_multi_step_simulation_forward` (:525): random target
  idx over denoising_step_list [1000,757,522]; from pure randn, no_grad
  x0/renoise chain through steps < max idx (SAME math as WanDMDLoop:
  pred_noise_to_pred_video + scheduler.add_noise on the shift-8 FlowMatch
  table); grad only on the final forward at the target step.
- `_dmd_forward` (:591): under no_grad — t = shift_timestep(randint(0,1000))
  clamp(min,max = ratios 0.02/0.98 x 1000); noise randn; noisy =
  add_noise(x0_student); critic pred -> x0_fake; teacher cond + uncond ->
  x0 CFG in DMD2 parameterization `cond + w*(cond-uncond)` (w=3.5 in the
  1.3B script); grad = (x0_fake - x0_real)/|x0_student - x0_real|.mean(),
  nan_to_num; dmd_loss = 0.5*MSE(x0.float(), (x0-grad).float().detach()).
- `faker_score_forward` (:671): rollout under no_grad; fresh t/noise draws;
  critic pred; loss = mean((pred_noise - (noise - x0_student))**2) in BF16
  (no .float() — unlike finetune loss!).
- `_build_distill_input_kwargs` (:486): hidden = noisy BTCHW->BCTHW permute,
  timestep passed LONG (not bf16); text dict = cond (batch embeds+mask) or
  uncond (negative_prompt_embeds encoded once at init ~:1092).
- ALL RNG here is GLOBAL torch RNG (randint/randn without generators),
  seeded set_random_seed(seed+rank) — capture must record draws in call
  order and the anchor replays them (monkeypatch torch.randn/randint during
  capture).
- Timesteps recorded as fp32 warped values via shift_timestep
  (training_utils.py:1136: t*shift/(1+(shift-1)t) scaled x1000).

## DiffusionNFT RL build facts (#24, read from train/methods/rl/diffusion_nft.py)

- managed_train_step (:168): sample epoch -> score rewards -> group
  advantages -> inner train -> `_update_old_model` (EMA old<-student,
  decay schedule :828). Student optimizer only (method-managed).
- `_training_timestep_loss` (:661): xt = (1-t)x0 + t*noise (noise from the
  SEEDED per-SP cuda_generator); old/reference predictions no_grad; student
  forward with grad; advantage clip/modes -> r in [0,1]; positive = beta*fwd
  + (1-beta)*old; implicit negative = (1+beta)*old - beta*fwd; per-sample
  weighted x0 MSEs (fp64 weight factors, clip 1e-5); policy = (r*pos +
  (1-r)*neg)/beta * adv_clip_max, mean; + kl_beta * MSE(fwd, ref).
- `_compute_advantages` (:496): group by PROMPT over the all-gathered batch;
  (r - mean)/(std_unbiased=False + 1e-4); repeated per train timestep.
- Rewards: build_multi_reward_scorer (pickscore/clipscore on main; the
  merged config `examples/train/configs/rl/wan/diffusion_nft_pick_clip.yaml`
  is single-frame RL: num_frames 1, num_latent_t 1, 25 sampling steps,
  beta .1, kl_beta 1e-4, lr 3e-5).
- Sampler: methods/rl/common/sampling.py DiffusionSampler (ode = flow Euler
  == our WanDenoiseLoop math; sde_reflow variant); prompt K-repeat sampler
  so groups share prompts.
- VideoAlign (phase A): PR #1476 branch `maint/pr1476-runtime-compat`
  @518aeab0b ONLY — vendor `train/methods/rl/rewards/videoalign.py` +
  `third_party/rl_rewards/VideoAlign/` (Qwen2-VL BT reward, MQ/VQ/TA,
  transformers-5 patches); checkpoint = local clone of KwaiVGI/VideoReward
  via VIDEOALIGN_CHECKPOINT_PATH. Gate: identical scores on fixed videos.
- Gate plan (phase B first): capture wrappers on DiffusionNFTMethod
  (_sample_epoch outputs, rewards, advantages, per-inner-step losses +
  recorded noise draws); our NFTStep replays through 3x WanModelFV.

## DiffusionNFT RL — measured facts (capture v3, cluster GB200, tfv 5.13)

- Roles load through `maybe_load_fsdp_model` with
  `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)` over **fp32
  storage**: forwards run on bf16-CAST params (RMSNorm weights and
  scale_shift_table included — measured by the block0 bisection probes;
  `p.dtype` probing only shows the fp32 STORAGE dtype, and FSDP2 params are
  DTensors that crash naive `.numpy()` hashing). AdamW (`eps=1e-8`) and the
  old-policy EMA operate on the fp32 storage — at world 1 exactly the
  `_MasterOpt` fp32-master/bf16-compute chain proven by the finetune/DMD2
  gates. Forwards autocast bf16 (`predict_noise`, a no-op on bf16 compute),
  BTCHW→BCTHW permutes at the boundary, mask passed but unused by the DiT.
- Effective grad accum = `gradient_accumulation_steps × num_train_timesteps`
  (gate config: 2×4=8) = exactly the inner calls per outer step ⇒ the
  optimizer fires ONCE at epoch end ⇒ all step-0 losses are pre-update and
  **bitwise repeatable** across identical-seed capture runs (proven 3×).
  Step-1 spread (one AdamW step from atomics-noisy backward): 2.76e-2 max.
- Inner loop shuffles samples AND per-sample timestep order via
  `cuda_generator` perms — capture records the DIRECT loss inputs instead
  (row_idx byte-matched against pre-shuffle items, timestep column, adv,
  xt-noise), so the anchor replays without RNG replication.
- `_return_decay` is a `@staticmethod(step, decay_type)` — a plain-function
  monkeypatch rebinds as an instance method and shifts the args.
- UPSTREAM FINDING (transformers 5.13): `CLIPModel.get_*_features` returns
  a ModelOutput whose `.pooler_output` holds the PROJECTED features —
  main's `PickScoreScorer` crashes (`.norm` on ModelOutput). Unwrap on the
  PickScore instance only: `CLIPModel.forward` internally consumes the new
  contract, so a class-level patch breaks `ClipScoreScorer`. Validation
  also fires at iteration 0 (`0 % every_steps == 0`).
- VideoAlign runtime needs `accelerate>=1.1` at import (transformers
  Trainer device setup) — pip-installed into the cluster venv.
- v2.1 pieces: `train/diffusion_nft.py` (NFTStep: plain AdamW, autocast
  _fwd, live-param EMA `update_old`, verbatim loss; `compute_advantages`,
  `return_decay`), anchor mode `train_anchor rl` (loss/embeds dtype
  detected from bf16-representability of the recorded fp32 arrays),
  `train/videoalign.py` + `rl_rewards/VideoAlign/` (byte-identical vendor,
  sha256-gated) + `gates/videoalign_anchor.py`.

## Dataloader order (#17 tail) — anchor.train-data-main PASS

- `train/data.py` reproduces main's map-style loader at world 1:
  `os.walk(realpath(root))` + realpath each file (HF snapshots: the sort
  key is the resolved BLOB path) + one global path sort; sampler =
  `randperm(total, manual_seed(seed))` truncated to a batch multiple and
  chunked; rows decode `np.frombuffer(fp32)` (main ignores the `_dtype`
  column); text embeds pad/crop to 512 with a 1/0 mask.
- Gate: first-5 batches vs the finetune capture — caption + latents hash
  (post num_latent_t slice, bf16-cast) + embeds hash, all exact.
