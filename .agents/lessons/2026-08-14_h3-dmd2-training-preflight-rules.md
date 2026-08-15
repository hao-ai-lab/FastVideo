---
date: 2026-08-14
experiment: feat/h3-dmd2-vsa — data-free DMD2 distillation of MiniMax-H3 (33B, video+audio) on NVL72 GB200
category: training
severity: critical
---

# Preflight Rules for Large-Model Distillation Runs (distilled from the H3 DMD2 v1–v5 lineage)

Every rule below is a real failure we shipped, diagnosed, and fixed across
five production launches. Rules are grouped; each states the check, the
failure it prevents, and where it is now enforced in code (if it is).

## Numerics

1. **Master weights must be fp32.** In-place optimizer steps on bf16 params
   round away updates below ~half an ulp of each weight's magnitude:
   O(1)-magnitude norm gains froze **bit-identically for 1000 steps**
   (205/211 tensors) at lr 2e-6, silently capping convergence. Enforced:
   `fastvideo/train/trainer.py::_verify_master_weight_precision` and
   `fastvideo/training/training_utils.py::verify_master_weight_precision`
   hard-fail at train start (opt-outs:
   `training.model.allow_low_precision_master_weights`,
   `FASTVIDEO_ALLOW_LOW_PRECISION_MASTER_WEIGHTS=1`).
2. **Optimizer second moments must be fp32 regardless of param dtype.**
   `torch.zeros_like(p)` inherits the starved dtype; bf16 `exp_avg_sq` loses
   g² increments below ~0.4% of its running magnitude. Enforced in
   `AdamWBeta1Zero` (fp32 state + fp32 update math for non-fp32 params).
3. **Do not copy inference precision defaults into training configs.**
   `PipelineConfig.dit_precision` defaults to bf16 (inference); training
   YAMLs that copy it inherit rule-1's failure. `dit_precision: fp32` costs
   only sharded-master memory — FSDP `param_dtype=bf16` keeps compute speed
   (measured: no step-time change at 33B).
4. **Post-hoc freeze check (cheap, mapping-free):** base checkpoints are
   stored bf16, so any fp32-updated tensor is almost surely no longer
   exactly bf16-representable. Census a checkpoint's norm params for
   `x != bf16(x)` fractions — near-zero fractions mean weights are not
   moving. Verified on the v5 lineage: at checkpoint-500 with fp32 masters,
   210/210 student norm tensors moved (median 99.9% of elements), vs
   205/211 bit-frozen after 1000 bf16 steps.

## Schedules and shifts

5. **Recompute every timestep-space constant through the model's actual
   shift map** (`sigma' = s*t / (1 + (s-1)t)`). The Wan-copied ladder
   `[1000,757,522]` under H3's video shift 12 produced sigma
   {1.0, 0.974, 0.929} — a near-no-op middle step and an 86% final jump.
   Re-spaced to `[1000,667,333]` (uniform base-t, the same design rule the
   Wan recipe uses pre-warp).
6. **Check supervision *density* in sigma space, not t space.** Uniform-t
   sampling under shift 12 put 57% of score-model draws at sigma_v > 0.9 and
   ~0% below 0.2. Fix: `method.score_timestep_shift` (sample uniform in
   shifted-sigma, invert to base-t). Verify any new sampler with a quantile
   census before launching.
7. **Shared timestep-ratio bounds hit modalities asymmetrically.** With
   min_ratio 0.02, video's supervision floor was sigma 0.197 while audio's
   was 0.057 (shift 12 vs 3). Derive per-modality sigma floors from the
   ratios; H3 uses 0.005.
8. **Know whether a step list is consumed in base-t or shifted space.**
   `warp_denoising_step` maps the list through the (video) scheduler's
   shifted grid; H3's adapter *also* shifts per modality internally —
   enabling warp double-shifts. Wan needs warp on; H3 needs it off.

## Packed multi-modality

9. **Never take a single mean loss over a packed multi-modality sequence.**
   H3 packs video:audio at ~270:1 elements; a packed mean silently trains
   only video. Compute per-modality losses via `modality_slices()` and log
   each stream (`*_video`, `*_audio`) separately.

## Distillation specifics

10. **Check whether the teacher is guidance-distilled before configuring
    CFG.** H3's released checkpoint rejects CFG at inference; correct
    setting is `real_score_guidance_scale: 1.0`, and DMD2 then skips the
    unconditional teacher forward entirely (free speedup).
11. **Read DMD scalars as game state, not quality.** Generator loss =
    normalized critic–teacher disagreement at student samples (healthy band
    ~0.2–1.0; it *rises* from ~0 as the critic specializes away from its
    teacher init). Critic flow-matching loss = tracking meter (falling =
    keeping up). The 10x-LR blowup signature: critic loss exploding (5.2)
    plus student grad-norms pinned above clip (1.8–2.5 vs 1.0) — a run that
    reports 10x LR but delivers ~4x, into a broken critic.
12. **Grad-norm vs clip tells you the LR you actually delivered.** Isolated
    spikes above clip that mean-revert are normal equilibration; sustained
    saturation means the configured LR is a lie.
13. **Sampler A/Bs need a teacher control and matched seeds.** The
    stochastic-vs-deterministic verdict on a student is meaningless without
    knowing the teacher's own behavior under both hop rules (teacher x0
    overshoot at sigma≈1, std 2.33, dominated ours). File-size heuristics
    on encoded video are not a quality signal — high bitrate can be noise.

## Run ops

14. **Requeue-safe by default:** `resume_from_checkpoint: latest` +
    `sbatch --requeue`. DCP restores optimizer LRs over the YAML — use
    `reset_lr_on_resume: true` for LR-change experiments.
15. **Budget checkpoint disk before launch:** save size × retention vs
    filesystem free. fp32 masters tripled saves to 741 GB (measured; student+critic params + fp32 second moments); lustre at 98%
    would have killed the run at its 4th save. `checkpoints_total_limit`
    is part of the launch math, not a detail.
16. **Verify W&B auth (`wandb.Api().viewer` with the job's $HOME) before
    every launch**; the tracker now degrades online→offline→none instead of
    crashing 32 GPUs on a rotated key. Confirm step-0 media actually
    uploaded — the uploader has died on the first video burst before.
17. **Unique `run_name` per launch attempt.** Three runs sharing one name
    made W&B show another run's config (the "DP=12" confusion).
18. **Cluster quirks (Slinky/GB200):** compute pods have no /home (export
    HOME=lustre in every sbatch; submit with a lustre --chdir); PENDing
    jobs are what make the autoscaler create nodes (warmup-ladder + race
    for multi-node); released nodes deregister in seconds; boot watchdog +
    NCCL env pins for the silent IB comm-init wedge; `scontrol release`
    after "launch failed requeued held"; Slurm requeues silently truncate
    stdout (check elapsed vs log).
19. **Exports and deletions:** cross-user hardlinks are blocked
    (fs.protected_hardlinks) — use `--link-base` symlink exports (~66 GB vs
    465 GB); always clear stale `*.safetensors.index.json` when rewriting a
    module dir (a leftover base index shadows the fresh weights); before any
    rm of run artifacts, verify no symlinks/hardlinks point into originals
    and that the target is regenerable.
