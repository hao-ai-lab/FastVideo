# TDM port handoff (branch `tdm-port`, internal)

- Workload: land and harden the modular TDM training method on this
  branch and integrate the validated recipe and measurement findings
  from the standalone TDM port. Internal only: no PR or draft PR until
  there is a quality-assurance story (user direction 2026-09-22). All
  runtime validation stays on one Kubernetes node with exactly four
  GPUs; no Modal or multi-node jobs.
- Issue: `hao-ai-lab/FastVideo#775` (`[Feature] TDM`) for provenance;
  the branch is no longer issue-named (renamed from `issue/775-tdm` to
  `tdm-port` on 2026-09-22).
- Current production-code tip: `6676ef6b1`, signed and synchronized with `origin/issue/775-tdm` before the handoff-only durability commit. It commits faithful same-trajectory reuse and the strict scheduler-interval lower bound atop the paper Eq. 11 pseudo-Huber support. Later branch-tip commits may update only this active handoff. Pre-rebase tip was `46166b3099cbb9fc7973d6306a0cc287b63a06eb`.
- Current stage: Distributed correctness, ordered critic-before-generator optimization, faithful `separate` intervals, negative-prompt conditioning, the 1D Gaussian reverse-KL oracle, a 1000-step two-time-scale run, paper pseudo-Huber, same-trajectory coupling, their combined 100-step control, effective critic batch 32, rank-16/rank-128/full-weight critic capacity, real-score conditioning, norm-matched critic weighting, the Wan rank-16 student reachability ceiling, the full-weight H3 critic/gradient positive control, the H3 critic weight-direction Phase-A screen, the H3 critic-convergence curve, and the H3 critic context-coverage run have all been tested. No acceptable trained checkpoint exists. A 2026-09-17 source audit against the official TDM reference (`Luo-Yihong/TDM@81b019f`) and paper found the port objective faithful; a decomposition of the retained diagnostic gradient data shows the critic-error component of the generator gradient is `1.0-2.2x` the teacher-student signal in every H3 interval and `0.42-1.34x` in Wan, so every negative generator-direction gate measured critic fit rather than the TDM objective. The H3 context-coverage run (`64` training trajectories, fresh tau/proposal-noise contexts every update, `384` updates at lr `2e-5`) showed coverage-bank fit equals held-out fit and training/coverage/held-out MSEs agree per interval, refuting the fixed-bank overfitting hypothesis: the critic plateau is a domain-independent bias floor. Intervals 0-2 (large student-teacher gap) improved substantially; intervals 3-7 are floor-limited because the released student is already close to the base teacher and the critic residual is comparable to the gap.
- Next steps (priority order): (1) run the paper-scaled joint H3 TDM run where the student moves and the gap grows (critic `2e-5`, effective batch >=32 to absorb gradient spikes, generator update interval 1, >=1000 generator updates) and judge it on distributional quality rather than paired same-seed MS-SSIM; (2) alternatively, if a testable positive control is required first, build one with a deliberately larger student-teacher gap at all intervals (partially trained or rank-limited student) instead of another frozen released-student schedule; (3) retain multi-prompt/seed and motion/prompt-alignment metrics before accepting a branch-trained checkpoint. Do not spend more compute on frozen-student critic schedules, coverage, or capacity, do not launch another 100-update TDM control from the negative gates, and do not open a PR before quality passes.
- User constraints: Do not run the implementation review/adjudication loop until opening a PR. All runtime validation and experiments must use one Kubernetes node with exactly four GPUs. Do not use Modal or multi-node jobs.
- Safety: Use a local pre-rebase safety ref, GPG-sign rewritten commits, and push only with `--force-with-lease`.
- Known rebase conflicts: Wan config, Wan DMD stage, distribution-matching exports, `ModelBase` prediction types/timestep handling, and trackers. Preserve current-main AnyFlow, joint video/audio typing, SwanLab, DMD2 bounds, Wan attention configuration, and validation behavior.
- Open validation gaps: no acceptable trained checkpoint; no branch-trained SSIM reference; no student-side supervised reachability ceiling has isolated rank-16 student capacity from the learned critic or the four-step compression target; and acceptance criteria still rely heavily on paired student-vs-teacher MS-SSIM. The sample-target oracle is diagnostic rather than the exact conditional expectation. The full-weight critic ceiling was function-step calibrated in BF16 and is negative under that representation, but should not be generalized to every possible full-weight precision/optimizer recipe.
- GitHub policy: use `gh` authenticated as `macthecadillac`; do not post comments or open a PR without a later explicit request.

## PR preparation resume (2026-09-25): fix-issue Stage 3

- User direction: prepare a PR for issue `#775` from branch `tdm-port` using
  the `fix-issue` skill. Stage 4 (actually creating the draft PR) still needs a
  separate explicit user request.
- Branch state: `tdm-port` was copied from the private `FastVideo-internal`
  repo to the public fork `macthecadillac/FastVideo:tdm-port` at `cc5746203`
  (identical SHA verified on both remotes). PR base is
  `hao-ai-lab/FastVideo:main`; merge-base is `37d06a832`, the branch is 76
  ahead / 26 behind current `upstream/main` (`e90be598e`), and a local
  `git merge-tree` check against `upstream/main` reports no conflicts. Diff vs
  merge-base is 61 files, +8580/-41. No PR exists for the branch.
- GitHub state re-checked this resume: issue 775 open/assigned to
  `macthecadillac` with only the two historic comments (no proposed fix); no
  open `775 OR TDM` PR. `gh` verified as `macthecadillac`.
- Worktree: `/Users/maclee/Documents/Code/FastVideo/.worktrees/tdm-port`
  (macOS sandbox refuses to create `.gitmodules` by name, so this worktree
  excludes that one upstream file via a skip-worktree sparse exclusion; it is
  not part of the PR diff and is never modified).
- Resume stage: implementation is complete through Phase 5; the
  review/adjudication loop (fix-issue Stage 3) has not been run and the
  pre-PR `pre-commit run --all-files` gate has not run. Next: append-only
  review/adjudication on committed code, pre-commit gate, then present the
  draft PR message. The open quality gap (no acceptable trained checkpoint,
  no branch-trained SSIM reference) must be stated in the PR body.
- Pre-commit blocker (local macOS sandbox): `pre-commit` is not installed and
  cannot be installed here. The sandbox blocks PyPI (`pypi.org`,
  `files.pythonhosted.org` time out through the local proxy) while GitHub is
  reachable; there is no Docker, no Modal CLI, and no internal pip mirror, and
  `~/.cache`/`/tmp` are not writable. DGX Spark SSH (`mac@23.125.121.179:61676`)
  also times out at banner exchange from this machine. Installing via uv was
  prepared inside the approved temp dir (managed CPython 3.12 + venv) and only
  the package download fails. The pre-commit gate therefore needs either an
  environment with PyPI access (e.g. the authoring Linux sandbox or DGX Spark
  Docker) or a sandbox allowlist entry for PyPI before this PR can be declared
  ready.
- Pre-commit gate executed on DGX Spark (Modal CLI exists locally but the
  sandbox cannot read `~/.modal.toml`, so per the user's fallback the gate ran
  on DGX in the approved dev image). Fresh clone of
  `macthecadillac/FastVideo:tdm-port` at `21a9b9ba9` under
  `/home/mac/tdm-port-precheck`; persistent hook cache
  `/home/mac/tdm-precommit-home`; full log on DGX at `/tmp/pc.log`
  (container `tdm-precommit`). Result: **FAIL**.
  - yapf modifies 3 branch files:
    `fastvideo/train/methods/distribution_matching/tdm.py`,
    `fastvideo/train/models/base.py`,
    `fastvideo/train/models/minimax_h3/minimax_h3.py`.
  - mypy reports 4 errors: `fastvideo/train/models/base.py:266` and `:267`
    (`float(Any | None)`), `tdm.py:376` (list append tuple type),
    `tdm.py:399` (assignment type).
  - ruff, codespell, pymarkdown, actionlint, check-filenames, and suggestion
    all pass.
  - The yapf diff is captured locally for reuse; these mechanical gate
    failures and the Stage 3 review findings are the inputs to the
    adjudicator/fixer before the gate is rerun.

## Port summary (2026-09-26): TDM in FastVideo, validated on Wan and H3

Scope for the first PR (user direction 2026-09-26): **TDM + H3 only**. Out of
scope and explicitly not done: TDM coverage for other model families
(Kandinsky5 is the only other family with a distribution-matching recipe, a
DMD2 QAT config) and TDM on the Fast-distilled checkpoints (FastWan/FastH3).
Every gate below trains LoRA adapters on the **base released checkpoints**
(`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, `MiniMaxAI/MiniMax-H3`) with the base
model as teacher.

### What the deliverable is

`TDMMethod` in FastVideo's modular trainer, modality-general, with:

- a regression **warmup** (`method.warmup_steps`; default 200 for Wan, 0
  otherwise) that fits the student to the guidance-combined teacher x0 at its
  own rollout states, with the critic gated out of optimizers, LR schedulers,
  and grad-clip targets;
- a progressive **step ladder** (`method.tdm_step_ladder`) whose strictly
  decreasing stages select the active denoising schedule and whose final stage
  drives validation/inference (`dmd_denoising_steps`);
- **reference-free acceptance**: frame sharpness/contrast and latent-cloud
  diversity, because paired same-noise metrics rank blurry students above
  coherent ones (three independent demonstrations);
- **joint video+audio** support: per-modality trajectory/context dictionaries,
  one joint transformer forward per rollout step, per-modality sigma grids and
  model-time conversion, losses summed per modality, and guidance-1
  short-circuiting for guidance-distilled H3 (no unconditional branch);
- **H3 video-sparse attention** in training, with `method.tdm_vsa_apply_to`
  (`"student"` default, `"all"` to make the critic and teacher sparse too).

### Implementation inventory

| File | Role |
|---|---|
| `fastvideo/train/methods/distribution_matching/tdm.py` | `TDMMethod`: warmup, ladder, modality-general trajectory/context, losses, `tdm_vsa_apply_to` |
| `fastvideo/train/models/base.py` | TDM contract (`tdm_modalities`, `tdm_clean_latents`, `tdm_sigma_grid`, `tdm_terminal_sigma`, `tdm_max_trajectory_label`, `tdm_sigma_to_model_timestep`, `tdm_predict_x0`, `tdm_initial_noise`); defaults reproduce video-only behavior |
| `fastvideo/train/models/minimax_h3/minimax_h3.py` | H3 joint overrides, LoRA enablement, VSA metadata (tile 64), `num_train_timesteps` = 1000 |
| `fastvideo/attention/backends/video_sparse_attn_h3.py` | bf16 Q/K/V cast at the tile-64 Triton kernel boundary |
| `fastvideo/train/callbacks/validation.py` | per-record validation seed; `decode_on_all_ranks` for every rank's own sample |
| `fastvideo/pipelines/pipeline_batch_info.py` | `DECODE_ON_ALL_RANKS_KEY` batch contract |
| `fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_decoding.py` | video and audio decode stages honour the decode flag |
| `docs/training/train_infra.md` | TDM parameter table + "TDM on MiniMax H3 (joint video + audio)" section |
| `tests/local_tests/tdm/` | suite, tools, k8s and Slurm runners, README |

Assets and configs:

- Wan TDM: `examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml`,
  `tdm_t2v_lora_overfit.yaml`, `tdm_t2v_lora_fixed_recipe.yaml`, plus
  `tdm_overfit_prompts.txt`, `tdm_overfit_validation.json`,
  `tdm_multiprompt_train_prompts.txt`, `tdm_multiprompt_validation.json`.
- H3 TDM: `examples/train/configs/distribution_matching/overfit_minimax_h3_t2va_tdm.yaml`,
  `tdm_h3_overfit_validation.json`.
- Tools: `tests/local_tests/tdm/tools/tdm_video_report.py`,
  `tdm_multiprompt_report.py`, metrics in
  `fastvideo/train/utils/tdm_metrics.py`.
- Tests: `test_tdm_upstream_parity.py`, `test_tdm_method_unit.py`,
  `test_tdm_warmup_and_ladder.py`, `test_tdm_metrics.py`,
  `test_tdm_scheduler_math.py`, `test_tdm_config_smoke.py`,
  `test_tdm_h3_joint.py`, `test_tdm_h3_config_smoke.py`,
  `test_jsonl_tracker.py`, plus H3 cases in
  `fastvideo/tests/train/callbacks/test_validation.py`. Total **122 passing**
  (74 in `tests/local_tests/tdm/`, 48 in the validation-callback file).
- Runners: Wan Kubernetes scripts under `tests/local_tests/tdm/k8s/` (kept
  for the recipes) and `tests/local_tests/tdm/slurm/{h3_tdm_vsa_run.sh,
  launch_h3_tdm_vsa.sh,h3_tdm_vsa.sbatch}` for H3.

### Validated results (all reference-free; one sample per arm where noted)

| Gate | Result |
|---|---|
| Phase 0 parity vs the upstream transcription | 4 tests; mutation check fails exactly the CFG-combination test |
| Phase 2 tool vs the standalone warmup-ladder | frame std `88.27` reproduced exactly; sharpness separates arms inversely to paired metrics |
| Phase 3 Wan 1.3B modular (200 steps, 4 GPUs) | treatment sharpness `15 -> 230` (step 0 -> 200), control collapses to `5.1` |
| Phase 3.1 multi-prompt + held-out (4 prompts, 4 seeds each) | within-caption spread `0.503` vs control `0.303` (1.66x); held-out sharpness `250.8` vs `42.7`; seeds produced distinct videos |
| 4D H3 joint gate (200 steps, 768x1344) | `std 58.4 / sharpness 131.9`, 124 frames, 32 kHz stereo track; sharp studio scene matching the prompt |
| 4E VSA matrix (200 steps, 480x832) | dense `std 55.7 / sharpness 43.2`; VSA 0.5 student-only `56.1 / 37.8`; VSA 0.9 all-roles `54.5 / 36.5`; all coherent, prompt-matching |
| Empty-frame guard | validation now writes **4 mp4s, 0 errors** per event (was 1 and 3) |

Two bugs worth naming: the H3 LoRA `predict_x0`/`forward_joint` model-time
convention (`1 - sigma`) and the VSA tile-64 fp32 Q/K/V kernel boundary
(`Both operands must be same dtype. Got bf16 and fp32`).

### Evidence

`artifacts/tdm-port/phase2/` (tool reproduction), `phase3/` (Wan video
reports, metrics, frames), `phase3-multiprompt/` (diversity reports + frames),
`phase4/` (H3 dense gate, VSA matrix, dense control, reports and frames).

### Commit trail (branch `tdm-port`, tip at the summary commit)

Phase 0-2: `620b74b34`, `0cd15a8d8`, `79eaa368f`, `f028aff75`, `74651bafc`.
Phase 3: `1958dc122`, `b88ad86e7`, `e6c574a9f`, `37b088d5b`, `49908120a`.
Phase 3.1: `d1ddd37d5`, `16ee46307`, `46c2e98ba`, `b79498e12`.
Phase 4 design and 4A-4B: `f4201de7a`, `fe37d8cf9`, `86a006a49`, `c20c795dc`,
`9f907bc5d`.
4C-4D: `adea8a4b8`, `253017549`, `ee84473ee`, `6544c8340`, `3b6bc8899`,
`ecf4f69e3`, `5ee8a0550`, `42d851777`, `4f43e5abe`.
4E and the VSA follow-ups: `f94c06961`, `03bfad2ad`, `805b79b05`, `909ed72bb`,
`916035256`, `8a696cbc9`, `4e89276a9`, `82384c8ea`, `f1caf84e8`, `20b735065`.
Phase 5 docs: `01cd031cb`, `b5a8bd62f`, `8142e9535`.

### Operating notes

- Runtime validation now goes through **Slurm** (user direction): one node,
  four GB200 GPUs, launched from the Slinky login node
  (`vlm-mal004@100.73.17.13`) with a detached `srun` because `sbatch` fails
  site-side (`user_env_retrieval_failed_requeued_held`). The dev image runs
  via Pyxis/enroot with `/mnt/lustre` bind-mounted at `/workspace`, so config
  paths resolve unchanged.
- H3 needs a complete model directory: the `vlm-mal004` snapshot is missing
  the `transformer_ref` folder its `model_index.json` declares, so the config
  points at the sibling overlay
  `.../models--MiniMaxAI--MiniMax-H3/snapshots/h3-tdm-overlay`.
- TDM builds **zero latents from the config**, so the trajectory canvas is a
  config override (`num_height`/`num_width`) and needs no new preprocessed
  asset; 480x832 and 768x1344 were both exercised.
- H3 in-process validation corrupts the autograd/CUDA stream state for the
  next backward, so gates validate once at the final step
  (`run_at_start: false`, `every_steps: <max_train_steps>`) or sample from
  checkpoints out of process.

### Limits and open items

- **No acceptable trained checkpoint and no branch-trained SSIM reference.**
  Every gate is a single-sample, 200-step overfit that proves the machinery and
  the recipe, not model quality. A scaled run (critic `2e-5`, effective batch
  >=32, generator interval 1, >=1000 generator updates, judged on
  distributional quality) is the missing quality story; at 480x832 that is
  roughly 8-9 h on the Slurm cluster.
- No supervised student-side reachability ceiling; the sample-target oracle is
  diagnostic rather than the exact conditional expectation.
- `pre-commit run --all-files` has not run (hooks unavailable in the authoring
  sandbox); the docs were hand-checked for non-ASCII, trailing whitespace, and
  long lines.
- PVC hygiene: `/workspace/issue-775/h3_model_teacher` is a broken early
  overlay safe to delete; `/workspace/run` is root-owned and unwritable from
  Slurm, so new runs go under `/workspace/vlm-mal004/tdm-port/runs`.
- Deferred tooling (low value, never needed): the step-count preflight sampler
  and a checkpoint rescorer from Phase 2.

## Phase 0 (2026-09-22): integration plan, branch rename, parity gate

User decisions (2026-09-22):

- Work stays internal on this branch; no PR or draft PR until there is a
  quality-assurance story. Branch renamed `issue/775-tdm` -> `tdm-port`
  (local ref renamed, pushed to the internal origin as `tdm-port`; the
  stale `fork/issue-775-tdm` tracking ref was removed; no GitHub fork
  branch with the old name exists).
- The regression warmup becomes the default for the TDM + Wan code path
  and must be documented.
- A cross-implementation parity gate is a prerequisite: the modular
  `TDMMethod` numbers must match the standalone/upstream-verified TDM
  math on identical inputs.

What the standalone port changed about this port's open problems
(evidence: `.agents/handoffs/tdm-standalone-{wan,h3}-handoff.md` on the
standalone branches and `artifacts/tdm-standalone-s10/`):

- **Warmup is load-bearing.** Warmup alone collapses visually (blurry
  orange blobs on every seed); TDM alone drifts out of the teacher cloud
  (this port's earlier overfit runs and the standalone full-weight
  S4-S8 grid); warmup + TDM yields coherent, sharp 4-step Wan samples.
- **Warmup target must be the CFG-combined teacher x0** (the real half
  of TDM's generator target), not the plain conditional teacher.
- **Progressive step ladder** (16 -> 8 -> 4) with weights-only stage
  init and fresh optimizers is what keeps each TDM stage inside its
  local basin.
- **Paired metrics are unusable as gates here.** Latent nearest-neighbour
  / MMD and paired MS-SSIM all rank the blurry no-guidance student above
  the coherent distilled one, in three independent demonstrations. Gates
  must be reference-free (coherence, sharpness, diversity,
  cross-prompt), with the cloud metrics recorded but not asserted.
- The critic-error-dominance diagnosis recorded in this handoff is
  complementary rather than contradictory: the regression warmup puts
  the student's states on-manifold so the critic residual stops
  dominating the generator signal. The earlier "positive control with a
  deliberately larger student-teacher gap" is inverted by the new
  evidence: the worked direction is to reduce the gap first, and the
  positive control is the warmup-only arm, which fails visually
  (artifacts/tdm-standalone-s10/warmup-ladder/).
- **H3 payload to reuse:** one frozen base plus `student`/`critic`
  adapters and an adapters-disabled teacher (mandatory at 33B); joint
  video+audio rollout and loss with per-modality shifts 12/3 and model
  time `1 - sigma`; guidance=1 specialization (no CFG branch); VSA-H3
  tile-64 Triton training recipe with tau 0.9 and `apply_to: all`;
  per-processor VSA state under activation checkpointing; and the fp32
  QK-norm -> bf16 kernel cast.

Phase plan (from the 2026-09-22 plan review):

0. Parity gate and branch hygiene (this section).
1. Warmup phase and step ladder inside `TDMMethod`; warmup default-on
   for the Wan path; stage chaining without optimizer-state carryover.
2. Reference-free measurement policy plus the standalone diagnostic
   tools (step sweep, cloud stats, perceptual rescore, warmup-only
   control).
3. Wan 1.3B end-to-end in the modular stack on one 4-GPU node.
4. H3 joint video+audio, shared adapters, then VSA.
5. Docs and examples; PRs only after quality passes.

### Phase 3.1 (2026-09-22): multi-prompt + held-out diversity follow-up

User-directed follow-up before Phase 4. Phase 3 could not measure sample
diversity because the validation set had one prompt; the standalone S10
found diversity contraction (~21 pct) on the train prompt with partial
held-out generalization and named multi-prompt training the standard fix.

Scope (user-approved): train the fixed recipe on four prompts and validate
on four train plus four held-out prompts with four repeated rows each, so
the reference-free report can separate within-caption seed spread from
cross-caption spread.

- Assets: `tdm_multiprompt_train_prompts.txt` (four object-centric train
  prompts) and `tdm_multiprompt_validation.json` (four train + four
  held-out captions, each repeated four times; the repeat makes the
  caption-to-filename mapping deterministic because the validation dataset
  shards rows across SP groups and pads to that degree).
- Tool: `tests/local_tests/tdm/tools/tdm_multiprompt_report.py` rebuilds the
  row mapping from the caption file and reports per-caption sharpness plus
  within-caption spread, cross-caption spread, and their ratio. Spread is
  the mean pairwise L2 between temporal-mean pooled frame descriptors, a
  reference-free proxy; paired metrics stay forensic.
- Runner: `tests/local_tests/tdm/k8s/run_wan_tdm_multiprompt.sh` builds the
  four-prompt text-only dataset on the pod (one prompt per shard), runs the
  treatment (warmup 100) and the warmup-only control (warmup 200) with
  `every_steps: 50`, then reports both arms.
- Pre-registered reading: the treatment should keep within-caption spread
  above the collapsing control while holding sharpness, cross-caption
  spread should exceed within-caption spread for the treatment, and the
  held-out captions should stay coherent (not far below the train
  captions). A control within-caption spread near zero with blurred
  sharpness is the expected conditional-mean collapse.

## Phase 4 (2026-09-22): H3 joint video+audio, shared adapters, then VSA

Scope (user direction 2026-09-22): bring the H3 plan's Phases 0-4 into the
modular trainer, then wire VSA. Acceptance is the reference-free policy, not
the H3 plan's older paired-cloud preregistration, because the standalone
work showed paired metrics rank the blurry arm above the coherent one.

### Recon facts that drive the design

- The modular H3 plugin (`fastvideo/train/models/minimax_h3/minimax_h3.py`)
  already returns the ordered `(-video_velocity, -audio_velocity)` pair from
  `predict_noise`, so the base conversion `x0 = x_t - sigma * pred_noise`
  yields H3's data-ward `x0 = x_t + sigma * v`. The sign bridge exists.
- `ModelBase.predict_x0` is video-only and raises on a tuple
  (`models/base.py:193`). The H3 plugin has no LoRA and no `predict_x0`, and
  its constructor takes no `lora` argument.
- H3 model time is exactly `1 - sigma` (no table lookup), and the sigma grid
  is the closed form `shift*u/(1+(shift-1)*u)` with `u = index /
  num_train_timesteps`; video shift 12, audio shift 3. This is the same
  static shifted-flow formula `TDMMethod._timestep_to_sigma` already has a
  fast path for, so the method's existing math generalizes if it is keyed by
  modality and each modality supplies its own scheduler.
- `enable_lora_training` supports `ReplicatedLinear` and H3's attention uses
  `to_q/to_k/to_v/to_out`, all in `DEFAULT_LORA_TARGET_MODULES`. LoRA is
  feasible without new plumbing. The modular trainer gives each role its own
  model instance, so student and critic each load a frozen base plus their
  own adapters and the teacher loads a frozen base without adapters (three
  bf16 copies, matching the standalone's proven memory profile).
- Every math helper in `TDMMethod` (`flow_effective_noise`, `flow_snr`,
  `flow_transition_to_noisier_sigma`, `_expand_sigma_for_latents`,
  `_mean_except_batch`) already operates on plain tensors, so a per-modality
  loop reuses them unchanged; only the trajectory/context containers and the
  loss reduction need to become modality-keyed.

### Design

Model side (H3 plugin):
1. Accept `lora` and call `_enable_lora_if_configured` (mirroring `WanModel`).
2. Add `tdm_modalities()` (`("video", "audio")`), `tdm_clean_latents(batch)`,
   `tdm_sigma_grid(index, modality)` from the closed form,
   `tdm_model_timestep(sigma, modality)` = `1 - sigma`, and
   `tdm_predict_x0(noisy, model_timesteps, batch, ...)` that runs one packed
   transformer forward and converts each modality with its own sigma. A
   `tdm_add_noise` mirror keeps the forward process per modality.
3. Default `ModelBase` implementations of these hooks return the video-only
   behavior so Wan is untouched.

Method side (`TDMMethod`):
4. Resolve `self._modalities` from the student and hold the per-modality
   trajectory/context as dicts keyed by modality (a one-key dict for Wan).
5. Loop the existing per-modality math over `self._modalities`, summing the
   warmup, fake-score, and generator losses across modalities, exactly like
   the standalone `_warmup_backward`/`_critic_backward`/`_generator_backward`
   sums. `tdm_step_ladder` and `tdm_denoising_steps` stay the shared integer
   grid; per-modality sigmas come from each modality's shift.
6. Keep the Wan path's numbers identical: the one-modality case must reduce
   to today's computation, and the Phase 3 tests must still pass.

Config and validation:
7. `overfit_minimax_h3_t2va_tdm.yaml` cloned from
   `overfit_minimax_h3_t2va.yaml`: LoRA student+critic (rank 16),
   `tdm_denoising_steps [999, 749, 500, 250]`, `generator_update_interval 1`,
   ladder, warmup, guidance 1 (no CFG branch). First gate at the cheaper
   `480x832x124` geometry before the production `768x1344x124`.
8. LoRA export/validation: the modular trainer saves DCP training state, so
   the H3 gate samples through the validation callback (now with per-record
   seeds) and the reference-free report, not a post-hoc checkpoint load.

Sub-steps and gates:
- 4A model-side H3 hooks + LoRA + unit tests on a CPU stub (no GPU).
- 4B method-side modality generalization; Wan tdm suite must stay green.
- 4C H3 TDM config + data + four-rank dry-run.
- 4D one-node/four-GPU H3 gate (480x832x124) with the reference-free report;
  VSA stays off for the first gate.
- 4E VSA wiring for H3 training (tile-64 Triton, tau 0.9, `apply_to: all`)
  then rerun the gate.

## Running log

- 2026-09-22: **Phase 4 started: design + model-side hooks done (4A), method
  generalization parked as WIP (4B).**
  - Recon confirmed the key facts: the modular H3 plugin already emits the
    negated velocity pair so the base `x0 = x_t - sigma * pred_noise` is
    H3's data-ward `x0 = x_t + sigma * v`; H3 model time is exactly
    `1 - sigma`; the H3 grid is the same static shifted-flow closed form the
    method already uses, with video shift 12 and audio shift 3;
    `enable_lora_training` supports `ReplicatedLinear` and H3's
    `to_q/to_k/to_v/to_out`, so LoRA needs no new plumbing.
  - 4A landed on `tdm-port`: `ModelBase` gained the TDM contract
    (`tdm_modalities`, `tdm_clean_latents`, `tdm_sigma_grid`,
    `tdm_terminal_sigma`, `tdm_max_trajectory_label`,
    `tdm_sigma_to_model_timestep`, `tdm_predict_x0`, `tdm_initial_noise`)
    with defaults that reproduce today's Wan behavior; the H3 plugin takes
    `lora` and calls `_enable_lora_if_configured`, and overrides the hooks
    for joint video+audio (`tdm_h3_sigma_grid`, `tdm_h3_model_timestep`,
    joint `tdm_predict_x0`). Commits `f4201de7a` (design),
    `fe37d8cf9` (hooks + LoRA). `test_tdm_h3_joint.py` (4 cases) plus the
    existing suites: **117 passed** on the held pod.
  - 4B (modality-general `TDMMethod`) is **done** on `tdm-port` at
    `c20c795dc`: per-modality trajectory/context dicts keyed by modality
    (one key for Wan), one joint forward per rollout step, label-space
    interval sampling, per-modality sigma grids and model-time conversion,
    summed warmup/fake-score/generator losses, and guidance-1
    short-circuiting for H3 (no unconditional branch). The Wan tests were
    migrated to the keyed API; `tests/local_tests/tdm/` is **70 passed** and
    the tdm + validation-callback suites are **117 passed**.
  - Parity lesson (important): a bitwise GPU metric comparison is **not** a
    valid parity gate on this stack. Two identical 5-step runs of the same
    code differed by `8.4e-2` on `grad_norm/student` at step 1 and grew
    chaotically, so the earlier "1.7e0"/"5.9e0" deltas against the archived
    Phase 3 metrics were nondeterminism, not a regression. The deterministic
    gate is the CPU parity suite (`test_tdm_upstream_parity.py`'s exact
    loss/gradient transcriptions plus the unit tests), which passes. The
    behavioral confirmation `wan-tdm-4b-confirm` (fixed recipe, treatment +
    control, validation every 25) reproduced the Phase 3 regime: treatment
    sharpness `15.0 -> 73 -> 114 -> 147 -> 322` over steps 0..200 while the
    control stays in the collapsing `~5-10` band.
  - Also removed: the parked `tdm-port-phase4-method-wip` branch, now folded
    into `tdm-port`. The one behavior-neutral change that mattered was the
    label-space candidate order; it must stay descending to match the
    schedule-sigma sampling it replaced.
  - 4C (H3 config + data + dry-run) is **done** at `adea8a4b8`:
    `examples/train/configs/distribution_matching/overfit_minimax_h3_t2va_tdm.yaml`
    (rank-16 LoRA student and critic over a frozen 33.12B base, frozen
    teacher, `real_score_guidance_scale: 1.0` with deliberately **no**
    `method.cfg_uncond`, warmup 50, 8->4 ladder, critic 1e-4 / generator
    2e-5), `tdm_h3_overfit_validation.json`, and
    `tests/local_tests/tdm/test_tdm_h3_config_smoke.py`. The H3 plugin now
    overrides `num_train_timesteps` to the 1000-label trajectory grid
    because the released scheduler exposes no training horizon and
    `DMD2Method._parse_score_timestep_bounds` needs it. A four-rank
    `--dry-run` completes `rc=0`; the tdm + validation-callback suites are
    **118 passed**.
  - H3 asset facts found the hard way (record these):
    - The `vlm-mal004` snapshot
      (`models--MiniMaxAI--MiniMax-H3/snapshots/42ed227e...`) is
      **incomplete**: its `model_index.json` declares `transformer_ref` but
      the subfolder is absent, so the modular loader rejects the directory.
      Use `/workspace/vlm-jileng/models/MiniMax-H3-teacher` instead (a
      complete dir: 62G transformer, 62G transformer_ref, 63G text encoder,
      `model_index.json` + `modular_model_index.json`).
    - The H3 transformer loads at **33.12B** parameters; rank-16 LoRA
      enables on **208** layers, matching `to_q/to_k/to_v/to_out`.
    - Training data: `/workspace/vlm-wlsaidhi/fastvideo/data/h3_overfit_t2va_0000007`
      (single row, `vae_latent [24, 37, 48, 84]`, `audio_latent [2, 32, 207]`,
      768x1344x124). Its caption is a long H3 multimodal description while
      the validation prompt is the standalone short prompt, so the gate
      checks that the joint path trains and stays coherent, not
      prompt-fidelity parity.
  - 4D feasibility answered by a 3-step smoke (`h3-tdm-smoke`, `rc=0`,
    no OOM): the joint H3 TDM loop trains at 768x1344 with `~96` s/step, so
    a 200-step arm is roughly **5.3 h** of wall clock. Two things had to be
    fixed first:
    1. The single-row H3 asset starved three of four ranks (`drop_last=True`
       plus rank sharding) and raised `generator raised StopIteration` from
       `trainer._iter_dataloader`. The gate therefore uses a 4-row replica
       at `/workspace/issue-775/h3_overfit_t2va_x4` (the same single sample
       duplicated once per rank, mirroring the Wan prompt x4 convention),
       which the config now points at.
    2. `MiniMaxH3Model.num_train_timesteps` returns the 1000-label grid, as
       above.
    The smoke also logs `FLASH_ATTN received torch.float32 inputs; casting to
    bfloat16`, so the H3 attention path runs a flash kernel and does not
    materialize dense attention at ~37k rows.
  - Next (4D proper): launch the 200-step H3 gate on the held four-GPU pod
    (validation every 25) once a ~5.3 h GPU window is approved, then judge
    video with `tdm_video_report.py` and add an audio-side reference-free
    check. If a shorter window is wanted, cut `max_train_steps` and keep the
    warmup-TDM split explicit.
  - 2026-09-23: the full 200-step gate is **running** as `h3-tdm-gate`
    (`tests/local_tests/tdm/k8s/run_h3_tdm_gate.sh`, launched 02:49:53Z).
    At ~96 s/step the train phase is ~5.3 h plus nine validations, so expect
    completion around 08:30Z. Judge it with `video_report.json` and add an
    audio-side check.
  - 2026-09-23: **4D is blocked on H3 validation, not on training.** Two
    model-asset and one runtime problem surfaced, in order:
    1. The `vlm-jileng/models/MiniMax-H3-teacher` dir is metadata-only for
       several components: its `text_encoder` has
       `model.safetensors.index.json` but no shards, and its `vae` has no
       safetensors. Training (which uses precomputed embeddings) never
       noticed; validation loads the pipeline and failed with
       `Cannot find any model weights` / `No safetensors files`.
    2. Fix: a **sibling overlay inside the mal004 cache** preserves the
       snapshot's relative blob symlinks. Built with
       `cp -rs <mal004 snapshot>/. <mal004 snapshots>/h3-tdm-overlay/`, then
       `transformer_ref -> /workspace/vlm-k1kong/models/MiniMax-H3/transformer_ref`
       (that copy has the 14 shards; vlm-jileng's does not). Verified shards:
       transformer 15, transformer_ref 15, text_encoder 15, vae 4. The config
       `init_from` now points at
       `/workspace/vlm-mal004/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/h3-tdm-overlay`.
    3. Remaining blocker: with the overlay the pipeline assembles and writes
       one validation video, then dies in
       `RuntimeError: CUDA driver error: invalid argument` from a
       torch.compile/inductor Triton kernel invoked through the LoRA linear
       (`fastvideo/layers/lora/linear.py:90` -> inductor -> triton launcher)
       during the validation rollout. Training alone does not hit it. Next
       things to try: set a writable `TRITON_CACHE_DIR` (the standalone H3
       gate did), disable torch.compile for validation, or run the first gate
       with validation sampled through a path that does not compile the LoRA
       linear.
  - Note: the 02:49Z gate attempt failed at 02:51Z on (1) and the 03:16Z
    attempt failed at 03:21Z on (2); both are recorded above. Do not treat
    the `h3-tdm-gate` status files as a completed gate until a run reaches
    `GATE_OK`.
  - Drifting-workstream findings (read-only, branches `feat/direct-drifting-generators`,
    `feat/h3-drifting-poc`, `analysis/h3-drifting-feasibility`):
    - The drifting H3 work post-trains the same **released 33B checkpoint** and
      its configs use `init_from: data/models/MiniMax-H3`. On the PVC that
      resolves to `/workspace/vlm-mal004/h3-drifting-h3-model-view-42ed227e`,
      a symlink farm whose members point at
      `/mnt/lustre/vlm-mal004/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227e...`.
      `/mnt/lustre` is the same Lustre as `/workspace` but is **not mounted in
      our pods**, so every member symlink of that view is broken here; the
      underlying weights are the very snapshot 4D already uses.
    - The view settles the missing-`transformer_ref` question: it maps
      `transformer_ref -> transformer` (the same weights), which is what
      satisfies the loader without a second 62 GB copy. Two pod-resolvable
      equivalents exist and both are legitimate: (a) the overlay already in
      the config (`h3-tdm-overlay`, `transformer_ref` from `vlm-k1kong` with
      real ref weights), or (b) the drifting convention of aliasing
      `transformer_ref` to `transformer`. Prefer (a) for a faithful ref
      component, (b) if a second copy is unwanted.
    - The drifting config also records the H3 geometry it validated:
      `num_latent_t: 37`, `num_height: 480`, `num_width: 832`,
      `num_frames: 124`, `sp_size: 4`, `hsdp_shard_dim: 4` on four GPUs and
      about **131 GB/GPU** at full resolution. That is the cheaper 480x832
      canvas the H3 plan wanted for the first gate and the source of the
      standalone 480x832 numbers.
  - 2026-09-23: **4D blocker is the in-process H3 validation path, not the
    TDM loop.** Three gate attempts with the overlay model dir:
    - attempt A (as shipped): validation wrote one video then `RuntimeError:
      CUDA driver error: invalid argument` from a torch.compile/inductor
      Triton kernel via the LoRA linear during the validation rollout.
    - attempt B (`TORCHDYNAMO_DISABLE=1`, writable `TRITON_CACHE_DIR`): the
      compile error disappeared but the *training* backward then failed with
      `RuntimeError: opt_ready_stream && opt_parent_stream INTERNAL ASSERT
      FAILED` (autograd engine). Validation again wrote one video first.
    - attempt C (`offload_training_state: false` as well): identical autograd
      stream assert, so returning the training state to the device is not the
      trigger. Running the H3 pipeline between training steps leaves the
      autograd/CUDA stream state inconsistent for the next backward. The
      3-step smoke with validation disabled passes, so training alone is fine.
  - 2026-09-23: **4D gate PASSED.** `h3-tdm-gate` completed `rc=0` with
    `GATE_OK` at 11:04:59Z: 200 steps of joint video+audio TDM on four GB200s
    with rank-16 LoRA student and critic over the frozen 33.12B base, at
    `~95-130` s/step (about 5.9 h). The final validation produced a
    **coherent, prompt-matching sample**: 124 frames at 768x1344 with a
    32 kHz stereo AAC track. Reference-free frame statistics at step 200:
    `std 58.36`, `sharpness 131.86` (for scale, the Wan treatment landed in
    the same coherent band while the Wan warmup-only control collapsed to
    `~5`). The frame is a lit studio scene with two red toy cars on a wooden
    turntable under softboxes, matching the validation prompt. Evidence:
    `artifacts/tdm-port/phase4/h3-gate-*` (report, status, frame).
  - Minor observation for the next gate: three of four ranks logged
    `ValueError: Validation media requires at least one video frame` and
    wrote no mp4 (non-fatal; the callback skips them). Only rank 0's sample
    survived, which is fine for a single-prompt overfit but should be
    understood before a multi-prompt gate.
  - Recommended next step for a full trajectory: run validation off, then
    sample from checkpoints in a separate process (resume-based sampler or
    `fastvideo/train/entrypoint/dcp_to_diffusers.py`), or fix the stream
    handling in the validation path. The cheaper 480x832 geometry remains a
    second lever: the drifting work validates H3 at 480x832, and
    `preprocess_minimax_h3_overfit.py` hardcodes 768x1344 with `crush-smol`
    source media, so a smaller asset is a small script change plus a
    preprocessing run.
  - 2026-09-23: **4E VSA wiring implemented; smoke blocked on a Triton dtype
    assert.** The H3 plugin now accepts `AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3`
    in addition to `TORCH_SDPA`, builds `MiniMaxH3VSAMetadataBuilder` metadata
    (tile 64, the validated Triton fwd/bwd geometry) in `prepare_batch` via the
    same `_h3_vsa_prefix_segments` helper the inference stage uses, and
    `predict_noise` now selects `batch.attn_metadata_vsa` when `attn_kind ==
    "vsa"` instead of forcing dense. `tdm_predict_x0` forwards `attn_kind`, so
    the existing TDM call sites already give the standalone's validated
    `apply_to: student` split (student sparse, critic and teacher dense). The
    dense path is unchanged: the builder returns `None` unless the backend is
    VSA-H3, and `tdm-port`'s suite is **71 passed**.
    - Smoke command: the H3 gate config plus
      `--models.student.attention_backend VIDEO_SPARSE_ATTN_H3 --vsa.sparsity 0.5
      --training.loop.max_train_steps 2`. It fails during the first forward
      inside a `fastvideo_kernel` Triton kernel with
      `triton.compiler.errors.CompilationError ... AssertionError: Both
      operands must be same dtype. Got bf16 and fp32`. This is the H3
      fp32/bf16 VSA-operand hazard the standalone port also had to fix.
    - Next things to try, in order: (1) match the inference stage's forward
      context exactly, which also passes `forward_batch=batch` into
      `set_forward_context` (the H3 training call passes only
      `current_timestep` and `attn_metadata`), in case the VSA impl reads the
      sparsity/dtype from the batch; (2) trace which operand stays fp32 (the
      pooled tile scores, the compression gate, or a QK norm) and cast it to
      bf16 at the boundary, as the standalone VSA port did for the fp32 QK
      norms; (3) confirm the tile-64 Triton route is selected rather than a
      256-tile CuTe fallback with different dtype rules.
    - Note the standalone's validated VSA is its own port
      (`tdm_standalone/tdm/vsa.py`), not the FastVideo
      `video_sparse_attn_h3` backend, so the modular route is new even though
      the recipe (tile 64, tau 0.5/0.9, `apply_to` student/all) transfers.
  - 2026-09-23: **blocked on GPU allocation, not on code.** The held
    `tdm-port-wan-r1` pod was reclaimed between sessions, and the replacement
    cannot start:
    - Every previously-used node (`10.0.133.7`, `10.0.140.245`, which have the
      dev image cached) reports `UnexpectedAdmissionError ... Requested: 4,
      Available: 0` for `nvidia.com/gpu`.
    - The scheduler keeps landing on `10.0.136.252`, which has four free GPUs
      but no cached image; its `ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev`
      pull has been "Pulling" for over 1.5 h of cumulative attempts with no
      `ErrImagePull`/`ImagePullBackOff`, so it looks slow rather than broken.
    - Worktree was also reclaimed by the sandbox reset; it was recreated from
      the pushed branch (`f94c06961`) and is disposable, so nothing is lost.
  - To resume 4E: get a four-GPU pod running (free a node that has the image
    cached, pre-pull the image on a free node, or wait out the pull on
    `10.0.136.252`), then re-run the VSA smoke
    (`--models.student.attention_backend VIDEO_SPARSE_ATTN_H3 --vsa.sparsity 0.5
    --training.loop.max_train_steps 2`) and work the dtype fix below.
  - 2026-09-23: **capacity sweep for the 4E smoke (all attempts failed).**
    Method and result, so this does not have to be redone:
    - `kubectl get node -o jsonpath='{.status.images[*].names}'` gives cached
      images per node. Fourteen GB200 nodes cache the exact
      `fastvideo-dev:py3.12-cuda13.0.0-latest` tag: `10.0.128.116`,
      `10.0.128.163`, `10.0.129.187`, `10.0.129.200`, `10.0.129.27`,
      `10.0.129.37`, `10.0.130.11`, `10.0.132.171`, `10.0.133.7`,
      `10.0.134.247`, `10.0.135.174`, `10.0.135.41`, `10.0.140.245`,
      `10.0.142.67`. Three more cache `py3.12-cuda13.0.0-sm100-latest`:
      `10.0.129.27`, `10.0.131.216`, `10.0.132.126`.
    - Pinning a 4-GPU pod to each of those sixteen nodes returned
      `UnexpectedAdmissionError ... Requested: 4, Available: 0` on every one,
      twice over. Re-probing all sixteen for a **single** GPU also found zero
      availability, so there is currently no free GPU on any node that has the
      image at all.
    - The only node with four free GPUs is `10.0.136.252` (healthy, no
      pressure, ~1 TB ephemeral, 593 GB of other cached images, but **no**
      FastVideo image). Its `Pulling` event has run 44+ minutes for one
      attempt and roughly 2.5 h cumulative across attempts with no
      `ErrImagePull`/`ImagePullBackOff`.
    - The main pod is left retrying on `10.0.136.252`; if that pull ever
      completes the four-GPU pod comes up on its own.
  - What would unblock it, in order of preference: (1) free one of the
    fourteen cached nodes (or tell me which job may be stopped); (2) pre-pull
    `fastvideo-dev:py3.12-cuda13.0.0-latest` onto `10.0.136.252` (or any free
    node) from a shell with registry access; (3) wait out the pull. A
    one-GPU node would be enough to find the 4E dtype bug with a small
    kernel-level test, but there is no such node free either.
  - 2026-09-23T23:28Z update: the `10.0.136.252` pull **completed** — the
    container reached `Started` — but the pod was then killed ~20 s later
    (BestEffort QoS, so first to go under node pressure). So the image is now
    cached on that node and the pull is no longer the blocker. The replacement
    pod is `Pending` with `FailedScheduling: 36 Insufficient nvidia.com/gpu`,
    i.e. **no GB200 node has four free GPUs at all** right now. Next resume
    should be quick once capacity appears, since at least one node now has the
    image.
  - 2026-09-24T05:00Z: **monitoring window ended without a pod.** The
    replacement pod sat `Pending` for 5h31m with 68 consecutive
    `FailedScheduling: 36 Insufficient nvidia.com/gpu` events — no GB200 node
    offered four free GPUs for the whole window. The pull is no longer a
    factor (the image completed and is cached on `10.0.136.252`), so once
    capacity appears the pod should schedule and start promptly. The pod is
    left `Pending` so the scheduler keeps trying. 4E remains where it was: the
    wiring is committed, the VSA smoke still needs to run to chase the
    bf16/fp32 dtype assert.
  - 2026-09-24T15:30Z: **capacity arrived; the 4E dtype bug is fixed and the
    VSA smoke now hits a memory wall instead.** Pod came up on `10.0.135.174`
    (a cached node).
    - Dtype root cause confirmed: the tile-64 Triton kernel
      (`fastvideo_kernel.block_sparse_attn`, `_attn_fwd_sparse`) does
      `tl.dot(p.to(bf16), v, acc)`, and Q/K/V arrived in **fp32** — the cast
      warning fired four times with `casting from torch.float32`. Fix:
      `MiniMaxH3VSAImpl.forward` casts Q/K/V to bf16 at the kernel boundary
      (commit `03bfad2ad`), matching the flash-attn dtype lesson. With that,
      Triton compiles and the run proceeds past the first forward.
    - New blocker, and it is real memory, not a bug: with VSA on at
      768x1344 the training step OOMs (`183.4 / 184.3 GiB` in use) even with
      `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The dense path fits
      at the same geometry, so VSA adds the last few GiB via its tile
      buffers/mask/scores.
    - `sp_size: 2` is **not** a quick lever: `--training.distributed.sp_size 2
      --training.distributed.hsdp_shard_dim 2` dies during FSDP setup with
      `IndexError: list index out of range` in the device-mesh
      `get_group(replicate_mesh_dim)`. The working 4-GPU mesh for this loader
      remains `sp_size 1 / hsdp_replicate 1 / hsdp_shard 4`.
    - Next lever (feasible, sources checked): build a **480x832** T2VA asset
      and rerun VSA there — the drifting-validated canvas, ~2.6x fewer video
      tokens. `fastvideo/pipelines/preprocess/preprocess_minimax_h3_overfit.py`
      hardcodes 768x1344, and the source media is present at
      `/workspace/vlm-aryan/fastvideo-h3-hybrid-training/data/crush-smol`, so
      this needs a small height/width parameterization plus pointing
      `MODEL_PATH` at the `h3-tdm-overlay` and one preprocessing run. Then
      replicate the row x4 for the four ranks, as with
      `h3_overfit_t2va_x4`.
  - 2026-09-24: **Jobs now go through Slurm** (user direction), not
    Kubernetes. Setup and the findings that matter:
    - Login: `ssh vlm-mal004@100.73.17.13` (Slinky login, key-based, works).
      Cluster: 20 GB200 nodes x4 GPUs, partitions `hpc-rack-2` (13 nodes),
      `hpc-rack-3` (7), `all` (20). All were fully allocated at recon time;
      Slurm queues rather than preempting, so a submitted job waits for
      capacity instead of being killed like the k8s BestEffort pod was.
    - Runtime: **Pyxis/enroot** `--container-image` is supported, so the same
      `ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest`
      image runs the job. There is no module system and no venv on Lustre.
    - Filesystem: `/mnt/lustre` on the login node is the `/workspace` of the
      k8s pods. Mounting it as `--container-mounts=/mnt/lustre:/workspace`
      keeps every absolute path in the shipped configs valid unchanged.
      Code is staged at `/mnt/lustre/vlm-mal004/tdm-port/code`.
    - **No 480x832 preprocessing is needed.** TDM calls
      `prepare_batch(latents_source="zeros")`, so the dataset's latent values
      and shapes are unused; the trajectory geometry comes from the training
      config. `--training.data.num_height 480 --training.data.num_width 832`
      therefore switches the canvas with no new asset, which removes the
      preprocess-script work the previous plan called for.
    - Runner: `tests/local_tests/tdm/slurm/h3_tdm_vsa.sbatch` (MODE=smoke|gate,
      HEIGHT/WIDTH/SPARSITY/MAX_STEPS/RUN_NAME env-driven). Job **11099**
      (VSA smoke, 2 steps, sparsity 0.5, 480x832) submitted 2026-09-24 and
      pending on `Resources`.
  - 2026-09-25: **Slurm path works and the VSA smoke passed.**
    - `sbatch` is unusable site-side: every submission (including a trivial
      `--wrap`) is requeued and held with
      `user_env_retrieval_failed_requeued_held`, and `--export=NONE` /
      `--get-user-env=none` do not help. The compute nodes run a minimal OS
      (no `/usr/bin/hostname`), so the site's env-retrieval helper cannot run
      there. **Workaround that works: a detached `srun` step**
      (`setsid nohup srun ... < /dev/null &`) from the login node, which is
      what `launch_h3_tdm_vsa.sh` does.
    - Other operational facts found the hard way: the compute-node home dir
      does not exist (`couldn't chdir to /home/vlm-mal004`), so run outputs
      must live on Lustre; `/workspace/run` was created by the Kubernetes
      jobs as root and is **not writable** from Slurm, so runs now go under
      `/workspace/vlm-mal004/tdm-port/runs/<name>`; the Pyxis/enroot image
      import takes ~10+ min per node and caches node-locally in
      `/tmp/enroot`.
    - VSA smoke (`h3-tdm-vsa-smoke`, 2 steps, 480x832, sparsity 0.5):
      **PASSED** at ~35 s/it, no dtype error and no OOM. That confirms both
      the kernel-boundary Q/K/V bf16 cast and the geometry lever.
    - VSA gate (`h3-tdm-vsa-gate`, 200 steps, same settings) launched and
      queued as Slurm job 11114; at ~35 s/it plus the final-step validation
      it is roughly 2.2 h once it starts.
  - 2026-09-26: **4E complete: H3 VSA training is wired and validated.**
    - VSA gate (`h3-tdm-vsa-gate`, 200 steps, 480x832, sparsity 0.5,
      student-only): completed. Sample is 124 frames at 480x832 with a
      32 kHz stereo track; frame statistics `std 56.12`, `sharpness 37.76`.
    - Same-geometry dense control (`h3-tdm-dense-480`, 200 steps,
      `TORCH_SDPA`): completed. `std 55.73`, `sharpness 43.15`. The
      kernel-boundary cast warning appears 4 times in the VSA log and 0
      times in the dense log, confirming the backends really differed.
      Step time at this geometry: VSA ~30 s/it vs dense ~26 s/it.
    - Reading: with sparsity 0.5 on the student only, VSA matches dense
      within ~12 pct sharpness and identical std, and the frames are
      visually equivalent (a red sports car on a wooden tabletop under
      studio lighting). Both sit far above the collapse band the Wan
      warmup-only control showed (~5). The H3 VSA training path is
      therefore usable; the open question is whether the aggressive corner
      (sparsity 0.9 with `apply_to: all`) holds quality, which needs the
      critic/teacher to run sparse too (a method-side `apply_to` knob).
    - Evidence: `artifacts/tdm-port/phase4/vsa-gate-480x832-*` and
      `dense-480x832-*` (reports, frames). The Slurm runners now take a
      BACKEND argument, so dense controls are `... h3-tdm-dense-480 TORCH_SDPA`.
  - 2026-09-26: **Phase 5 docs done.** `docs/training/train_infra.md` gains a
    "TDM on MiniMax H3 (joint video + audio)" subsection: the per-modality
    contract, the H3 sigma grid (shifts 12/3), `1 - sigma` model time, the
    data-ward velocity sign bridge, summed per-modality losses, the
    guidance-1/no-`cfg_uncond` rule, the standalone-derived LRs (critic 1e-4 /
    generator 2e-5), the geometry/memory levers (zeros latents mean the canvas
    is a config override; VSA at 768x1344 exceeds 184 GiB), VSA usage and the
    measured sparsity-0.5-vs-dense result, and the in-process validation
    stream-state caveat. The TDM parameter table now lists `warmup_steps` and
    `tdm_step_ladder`. The local-test README
    (`tests/local_tests/tdm/README.md`) drops the stale Modal/Kubernetes
    statements for the Slurm constraint, lists the H3 joint and H3 config
    tests, and points at the docs page for behavior.
  - Note for a follow-up: `pre-commit` could not run in this environment
    (not installed locally, and `pipx` cannot write under the sandbox), so the
    docs commits were checked by hand for non-ASCII characters, trailing
    whitespace, and over-long lines instead. Run
    `pre-commit run --files docs/training/train_infra.md tests/local_tests/tdm/README.md`
    where the hooks are available before any PR.
  - 2026-09-26: **`apply_to: all` knob added; the sparsity-0.9 gate is
    queued.** `method.tdm_vsa_apply_to` (`"student"` default, `"all"` optional)
    routes the critic and teacher through the sparse metadata too; the config
    must then set their `attention_backend` to the sparse backend. Three new
    unit tests pin the per-role selection (default student-only, `all`, and
    rejection of an unknown scope); the TDM suite is **74 passed**. Docs and
    the parameter table describe the knob. Gate
    `h3-tdm-vsa-t90-all` (200 steps, 480x832, sparsity 0.9, all roles sparse)
    queued as Slurm job 11148 for comparison against the sparsity-0.5
    student-only and dense controls.
  - 2026-09-26: **VSA matrix complete at 480x832.** All three arms are 200-step
    overfits with the final-step validation; one sample per arm:
    | arm | std | sharpness |
    |---|---|---|
    | dense (`TORCH_SDPA` control) | 55.73 | 43.15 |
    | VSA sparsity 0.5, student-only | 56.12 | 37.76 |
    | VSA sparsity 0.9, `tdm_vsa_apply_to: all` | 54.49 | 36.53 |
    All three are coherent, prompt-matching joint samples (124 frames at
    480x832 with a 32 kHz stereo track); the aggressive corner is visibly
    hazier around the subject, matching its lower sharpness. So the sparse
    recipes hold the dense band to within about 15 percent sharpness. Evidence:
    `artifacts/tdm-port/phase4/{dense,vsa-gate,vsa-t90-all}-480x832-*`.
  - 2026-09-26: **empty-frame validation guard fixed.** Root cause:
    `MiniMaxH3VideoDecodingStage` and `MiniMaxH3AudioDecodingStage` default to
    global-rank-zero output ownership unless the sequence-parallel decode path
    is active, so with `sp_size: 1` ranks 1-3 held placeholders. The
    validation callback's frame guard then logged `Validation media requires
    at least one video frame` and skipped them, which is why every H3 gate
    produced 1 mp4 instead of 4 - and a multi-prompt validation would have
    silently lost three quarters of its prompts. Fix: a generic
    `DECODE_ON_ALL_RANKS_KEY` batch flag in `pipeline_batch_info`, honoured by
    both H3 decode stages and set by the validation callback. Check run
    `h3-validate-ranks` (2 steps, validation at the final step):
    **4 mp4s, 0 guard errors** (previously 1 and 3). Suites: **122 passed**,
    including a new callback test pinning the flag.
  - Remaining: the eventual PR (only once a QA story exists, per the standing
    direction) and a pre-commit sweep where the hooks are available.

- 2026-09-22: Branch renamed to `tdm-port` and pushed to the internal
  origin; stale `fork/issue-775-tdm` ref removed. Phase 0 plan recorded
  above. Next: parity gate.
- 2026-09-22: **Phase 0 parity gate done.**
  `tests/local_tests/tdm/test_tdm_upstream_parity.py` added: it pins the
  modular method against the same upstream transcription the standalone
  port uses. Four tests cover (1) the assembled context's noise
  identities (`eps_source`, intermediate/target states, transition beta,
  mixed-noise reconstruction, scheduler sigma membership), (2) the
  critic loss composition (clipped SNR 5.0, importance
  `exp(0.5(||proposal||^2 - ||mixed||^2))` clamped at 10.0) with
  parameter gradients, (3) the generator composition (stop-gradient
  boundaries, CFG teacher target at 4.5, per-sample delta
  normalization) with parameter gradients, and (4) the default
  configuration being the upstream reference mode. The generator test
  also asserts that the generator phase re-derives the critic context's
  target state exactly (the trajectory-reuse invariant).
  - Results on the held GB200 pod: 4/4 parity tests pass; full
    `tests/local_tests/tdm/` suite 49 passed. Mutation check: replacing
    the CFG combination with the plain conditional teacher
    (`real_cfg_x0 = real_cond_x0`) fails exactly the generator parity
    test, confirming the transcription is discriminating.
  - Environment note: the suite imports the Triton driver chain
    (`fastvideo.attention`), so it needs a GPU pod; the CPU audit pod
    cannot collect it (`RuntimeError: 0 active drivers`). Stage the tree
    including `examples/train` or the config-smoke tests fail on missing
    YAML paths.
  - Next: Phase 1 - warmup phase and step ladder inside `TDMMethod`,
    warmup default-on for the Wan path, documented.
- 2026-09-22: **Phase 1 done.**
  - `method.warmup_steps` (default `200` for Wan-family students via
    `_model_family()`, `0` otherwise; explicit override wins, negative
    rejected). During warmup the student regresses onto the CFG-combined
    teacher x0 at its own rollout states; the critic takes no updates and
    is excluded from `get_optimizers` / `get_lr_schedulers` /
    `get_grad_clip_targets`, and its optimizer state stays empty.
    Metrics: `tdm/warmup`, `tdm/warmup/loss`, `tdm/warmup/guidance`,
    source timestep/sigma/trajectory-index, plus `tdm/warmup_steps` and
    `tdm/step_ladder_stage` on every step.
  - `method.tdm_step_ladder`: staged `denoising_steps` with
    `until_iteration` boundaries (strictly decreasing step counts,
    strictly increasing boundaries, only the final stage unbounded).
    The active stage is selected per iteration and the denoising
    assets/sigmas are recomputed per stage. Validation and inference
    adopt the final stage because `method_config["dmd_denoising_steps"]`
    is set from it.
  - The shipped example config carries `warmup_steps: 200` plus a
    commented 8 -> 4 ladder; the config smoke test pins the warmup value.
  - Docs: `tests/local_tests/tdm/README.md` gains "Warmup And Step
    Ladder" (standalone evidence, defaults, ladder rules, paired-metric
    caveat); its validation section now records the one-node/four-GPU
    constraint instead of the stale Modal instruction.
  - Tests: `test_tdm_warmup_and_ladder.py` (11 cases): Wan default,
    explicit override and negative rejection, student-only gating,
    CFG-teacher regression target with gradients, ladder stage
    resolution/sigma recomputation, and six malformed-ladder
    rejections. Full `tests/local_tests/tdm/` suite: **60 passed** on the
    held GB200 pod. Mutation check: replacing the warmup CFG combination
    with the plain conditional teacher fails exactly
    `test_warmup_loss_matches_cfg_teacher_target`.
  - Scope note: stage chaining is implemented as an *in-run* ladder, so
    no framework-level weights-only init was needed; the existing
    `dcp_to_diffusers` + `transformer_override_safetensor` path remains
    the option for cross-run chaining with fresh optimizer state (not
    implemented). Optimizer state carries across ladder stages whereas
    the standalone recipe reset it per stage; revisit in Phase 3 if the
    validation run shows it matters.
  - Next: Phase 2 - reference-free measurement policy and the standalone
    diagnostic tools.
- 2026-09-22: **Phase 2 done (measurement policy + first tools).**
  - `fastvideo/train/utils/tdm_metrics.py`: reference-free metrics
    (frame statistics with a gradient-sharpness blur detector, latent-cloud
    diversity, teacher-cloud overlap and the median-squared-distance
    bandwidth convention) plus forensic-only
    `nearest_neighbour_relative_mse` and optional `paired_ms_ssim`.
    `test_tdm_metrics.py` (4 cases) pins the sharpness ordering,
    mode-tightening detection, overlap behaviour, and the documented
    paired-metric inversion.
  - `tests/local_tests/tdm/tools/tdm_video_report.py`: scans a run's
    `*/samples/step-*` validation videos, emits frame statistics per
    video plus optional forensic paired MS-SSIM, writes JSON.
  - Phase 2 gate result: the tool reproduces the standalone warmup-ladder
    frame stat exactly (step-400 student std `88.27` equals the standalone
    `frame_stats.json`) and separates the arms on sharpness -
    warmup-only `22-64`, teacher `90.5`, TDM ladder `192-401` - the exact
    inverse of the paired-metric ranking (warmup-only `nn_rel_mse 0.606`
    "in bounds" versus TDM `1.350` "out"). Evidence:
    `artifacts/tdm-port/phase2/`.
  - Warmup-phase control tests appended to
    `test_tdm_warmup_and_ladder.py` (warmup-only never touches the critic;
    the TDM phase updates both roles). Full TDM suite: **62 passed**.
  - Deferred to Phase 3 and documented in the README: the step-count
    preflight sampler and a checkpoint rescorer, because both need real
    checkpoints/weights and the modular LoRA export path resolved. The
    standalone equivalents already produced the S10 numbers.
  - Commits pushed: `79eaa368f` (metrics), `f028aff75` (tool + controls).
  - Next: Phase 3 - Wan 1.3B end-to-end in the modular stack on one
    four-GPU Kubernetes node.
- 2026-09-22: **Phase 3 first run (shipped learning rates) is a no-op regime.**
  - The four-GPU pod `tdm-port-wan-r1` scheduled immediately. The new
    `tdm_t2v_lora_fixed_recipe.yaml` (warmup 100 + 8->4 ladder, 200 steps)
    passed a four-rank dry-run, and the on-PVC one-prompt text-only
    dataset from the earlier work
    (`/workspace/issue-775/tdm-overfit-447ebf2-r4/data/tdm_t2v_overfit_text_only`)
    was reused.
  - `wan-tdm-fixed-recipe` (treatment: warmup 100 then TDM 100; control:
    warmup 200) completed rc=0, ~34 min per arm, with validation videos
    every 25 steps. The report tool needed a fix to accept the modular
    `validation_step_*_rank_*_video_*.mp4` layout (committed).
  - The recipe plumbing is correct: `tdm/warmup` is 1.0 through step 99,
    `tdm/step_ladder_stage` switches 0 -> 1 at step 100, warmup loss
    3.36 -> 2.38 (grad norms ~1), TDM generator loss ~0.33, critic loss
    ~7e-4, critic grad norms ~5e-4.
  - But **both arms are visually and metrically identical to the base**:
    a blurred blob, frame sharpness 15.0 -> 10.1 in both, std ~30. The
    cause is the shipped overfit recipe's learning rates - student 2e-6
    and fake-score 8e-6, 50-100x below the standalone-validated 1e-4 -
    which move the adapters by roughly 1e-4 over 200 steps. This is the
    same no-op regime as standalone S9.
  - Rerun launched with student 1e-4 / fake-score 1e-4 (the runner now
    takes `STUDENT_LR` / `FAKE_SCORE_LR`), everything else identical.
  - Evidence: `artifacts/tdm-port/phase3/` (both video reports, status,
    frames at steps 0/100/200 for both arms).
- 2026-09-22: **Phase 3 gate met: the modular stack reproduces S10/S11.**
  - Rerun `wan-tdm-fixed-recipe-lr` (student `1e-4` / fake-score `1e-4`,
    warmup 100 + 8->4 ladder; control = warmup 200 with everything else
    identical) completed rc=0 in ~35 min per arm.
  - Frame statistics per validation step (mean over the four rank
    videos), treatment | control: step 0 `std 30.1 / sharpness 15.0` both;
    step 75 `65.1 / 12.0` versus `48.5 / 3.5`; step 100 `76.1 / 40.0`
    versus `66.3 / 8.4`; step 125 `102.1 / 188.8` versus `57.3 / 14.2`;
    step 200 `91.5 / 117.0` versus `48.7 / 6.4`.
  - Visuals: the treatment's step-200 sample is a sharp red sports car
    with the circular motion trail matching the prompt; the control is a
    structureless red smear (the conditional-mean collapse). The arms are
    identical at step 0 and diverge materially from the TDM phase onward
    (the step-100 validation includes the first TDM update); a ~1 pct
    early divergence at steps 25-75 is immaterial.
  - Shipped configs updated with the evidence: the two diagnostic TDM
    configs now ship student `1e-4` / fake-score `1e-4`; the
    production-shaped `tdm_t2v_lora.yaml` keeps the DMD2-inherited `2e-6`
    with a warning comment because it has not been re-validated at the
    higher rate. The README documents the regime and its measurement.
  - Deferred and documented: the multi-prompt diversity run and the
    held-out-prompt check, because the one-prompt validation dataset
    limits this phase to the coherence and sharpness gates.
  - Evidence: `artifacts/tdm-port/phase3/` (LR video reports, per-step
    tables, tracker metrics, stage log, frames at steps 0/100/125/200).
  - Next: Phase 4 (H3 joint video+audio, shared adapters, then VSA) or a
    short Phase 3 follow-up (multi-prompt diversity) per user direction.
- 2026-09-22: **Phase 3.1 started (multi-prompt + held-out diversity).**
  User chose the multi-prompt-training scope with four seeds per prompt.
  Added the four-prompt train list, the four-train/four-held-out validation
  file (four repeats each), the reference-free multi-prompt report tool,
  and the pod runner (preprocess four prompt shards, two arms, two
  reports). Plan and pre-registered reading in the Phase 3.1 section.
  Next: launch the held four-GPU pod and run.
- 2026-09-22: **Phase 3.1 first run exposes a fixed-seed validation flaw.**
  Both arms completed rc=0 (`wan-tdm-multiprompt`, 16:55Z to 18:04Z) but
  every caption's four repeated rows produced byte-identical videos, so
  `within_spread` was exactly `0.0` and the seed-diversity axis was not
  measured. Root cause: the Wan DMD latent stage seeds the initial noise
  from `batch.seed` (`fastvideo/pipelines/stages/latent_preparation.py`,
  `_arch_invariant_randn`), and the validation callback wrote the training
  seed into every record, so each caption had exactly one noise draw.
  Repeated rows cannot create seed diversity; `num_videos_per_prompt` also
  reuses the fixed seed.
  - Fix: `ValidationCallback._prepare_validation_batch` now honors an
    optional per-record `seed` (falling back to the training seed), the
    validation file carries four explicit seeds per caption, and
    `test_prepare_validation_batch_uses_record_seed` pins the behavior.
    Backwards compatible: records without a `seed` are unchanged.
  - Sharpness results from the first run are still meaningful and
    reported below; the rerun `wan-tdm-multiprompt-seed` repeats both arms
    with the per-row seeds.
  - First-run frame sharpness (train | held-out), treatment at steps
    0/50/100/150/200: `15.0 | 35.9`, `26.5 | 25.5`, `34.6 | 47.1`,
    `39.5 | 131.7`, `95.0 | 273.9`; control: `15.0 | 35.9`, `24.2 | 25.4`,
    `10.2 | 27.6`, `10.4 | 44.0`, `3.0 | 17.0`. The treatment separates
    from the collapsing control on both train and held-out prompts, so the
    fixed recipe does generalize to unseen prompts in the coherence
    sense; held-out sharpness runs higher because those prompts are more
    textured.
  - Tests on the pod after the fix: `tests/local_tests/tdm/` plus
    `fastvideo/tests/train/callbacks/test_validation.py` = 113 passed.
- 2026-09-22: **Phase 3.1 done: diversity recovery and held-out
  generalization both hold.**
  Rerun `wan-tdm-multiprompt-seed` (four train prompts, four seeds per
  caption, treatment warmup 100 then TDM 100, control warmup 200)
  completed rc=0, 18:22Z to 19:31Z.
  - Per-step report, treatment | control. Within-caption seed spread:
    `0.262 | 0.262` (step 0), `0.354 | 0.350` (50), `0.405 | 0.330` (100),
    `0.466 | 0.322` (150), `0.503 | 0.303` (200). Cross-caption spread:
    `0.342 | 0.342`, `0.572 | 0.570`, `0.627 | 0.553`, `0.678 | 0.551`,
    `0.748 | 0.486`. The treatment nearly doubles its seed spread while
    the control stalls and then tightens, a `1.66x` separation at step
    200.
  - Frame sharpness, train | held-out: treatment `18.9 | 55.5` ->
    `15.4 | 52.8` -> `20.5 | 72.9` -> `129.9 | 190.6` -> `229.8 | 250.8`;
    control `18.9 | 55.5` -> `16.3 | 56.1` -> `10.4 | 52.1` ->
    `12.5 | 88.4` -> `5.1 | 42.7`.
  - Per-caption step 200: every treatment caption is coherent
    (`112-319` sharpness, within-caption spread `0.41-0.59`); the control's
    four train captions collapse (`2.3-6.2`) while its four held-out
    captions stay partly coherent (`19.7-103.2`) because it never
    regressed those prompts.
  - Visual check (frames under the local
    `artifacts/tdm-port/phase3-multiprompt/frames/`): the treatment renders
    a crisp red sports car on the white tabletop and, on an unseen prompt,
    a sharp golden retriever puppy in tall grass; the control renders a
    blurred smear on the train prompt and a soft base-model-like dog on
    the held-out one.
  - Reading: the multi-prompt fix named by the standalone S10 works. The
    fixed recipe no longer contracts on the training prompt; it grows seed
    diversity and generalizes to unseen prompts in the coherence sense.
    The warmup-only collapse is prompt-specific, which is why its held-out
    sharpness stays above its train sharpness.
  - Evidence: `artifacts/tdm-port/phase3-multiprompt/` (both reports, stage
    log, per-arm report logs, eight frames); run root
    `/workspace/run/tdm-port/wan-tdm-multiprompt-seed`.
  - Commits: `d1ddd37d5` (assets), `16ee46307` (runner wait fix),
    `46c2e98ba` (per-record validation seed).
  - Next: Phase 4 (H3 joint video+audio, shared adapters, then VSA).

- 2026-09-14: User approved the rebase and overfitting plan. Read the `launch-experiment` and `evaluate-video-quality` skills. The skill's legacy experiment-journal requirement conflicts with current repository guidance against `.agents` experiment journals, so experiment state will be maintained in this mandatory handoff instead.

- 2026-09-14: Verified `gh` identity is `macthecadillac`. Re-checked issue #775: open, assigned to `macthecadillac`, two existing comments with no proposed fix. Re-checked open PRs: no PR for TDM/issue 775.
- 2026-09-14: Fetched `upstream/main` at `bfc9c017977d1f428d43b6e580a0ff20503d442b` and `origin/issue/775-tdm` at `46166b3099cbb9fc7973d6306a0cc287b63a06eb`. Merge base remains `fc02a9ce8ef0df261366e94454de6836036350e6`; branch is 125 commits ahead and 179 behind.
- 2026-09-14: The 125-commit range contains many tracked handoff, diagnostic-record, and temporary K8s orchestration commits that cancel out in the final tree. Rebase strategy is to preserve the old tip with a local safety ref and reconstruct the effective net TDM changes on current main in focused signed commits, omitting removed agent artifacts.

- 2026-09-14: Created local safety ref refs/backup/issue-775-tdm-pre-rebase-20260914 at the old tip and dedicated worktree /tmp/fastvideo-worktrees/issue-775-tdm.
- 2026-09-14: Reconstructed the net TDM tree on upstream/main via a squash rebase. The only conflicts were the five predicted files. Kept current compatibility shims/generic denoising stage and relocated TDM changes to fastvideo/models/wan/pipeline_config.py and fastvideo/pipelines/basic/wan/stages/dmd.py.
- 2026-09-14: Preserved current AnyFlow/AnyFlowPretrain/StreamingLongTuning exports, joint video/audio prediction typing, SwanLab/W&B behavior, role-local attention configuration, and current validation code. Added TDM timestep expansion, TDM export, JSONL alongside current trackers, flow-shift inheritance, DMD ODE/scheduler-space support, and checkpoint hardening.
- 2026-09-14: Updated TDM tests from stale generic DMD/config imports to current family-local Wan paths and corrected tracker documentation to retain SwanLab. No unmerged paths remain; staged and unstaged git diff --check pass.

- 2026-09-14: Created and pushed signed rebase commit `1c1574de554cda3160fafb961546fded2d86aa72` (`[feat]: add Wan TDM training`) to `origin/issue/775-tdm` with an exact force-with-lease against old tip `46166b3099cbb9fc7973d6306a0cc287b63a06eb`. Verified the remote ref through `gh`; branch is one commit ahead and zero behind upstream main. No tracked `.agents` files and no PR exist.
- 2026-09-14: The temporary worktree disappeared during an environment recycle, but the signed/pushed rebase and local safety ref remained. Recreated `/tmp/fastvideo-worktrees/issue-775-tdm` from `issue/775-tdm` at `1c1574de5`.
- 2026-09-14: Prepared `tdm_t2v_lora_overfit.yaml`, a four-line identical prompt input, one-prompt fixed validation JSON, and a config regression. Recipe uses one node/four GPUs, HSDP shard 4, text-only simulate mode, generator interval 1, `next_step`, no random midpoint, zero unconditional text, JSONL-only tracking, 100 steps, and 25-step checkpoints. Runtime validation is disabled so checkpoint sampling can be run separately with a fixed seed. Static `git diff --check` passes. No local tests were run.

- 2026-09-14: Committed the overfit recipe as signed commit `2982ecf526d427e155793c4d4db01e266ee3da24` (`[test]: add Wan TDM overfit diagnostic`) and immediately pushed it. GPG signature is good; the GitHub branch ref matches exactly and the worktree is clean.
- 2026-09-14: Kubernetes access recovered. Context `default`, namespace `vllm`, user `vlm-mal004`; all GPU nodes expose exactly four `NVIDIA-GB200` GPUs as `BM.GPU.GB200.4`. Confirmed `lustre-pvc-vllm`, `hf-token`, pod creation, and pod-log access. No active issue-775 pods/jobs existed.
- 2026-09-14: Rendered `/tmp/issue-775-k8s/tdm-overfit-2982ecf-r1-{run.sh,pod.yaml}` and created ConfigMap `tdm-overfit-2982ecf-r1-script`. Server dry-run passed. The pod requests exactly four GPUs and pins one GB200 node. Its ordered gates are targeted TDM/DMD2/checkpoint tests, four-rank NCCL, four parallel one-prompt text-only preprocess workers with exact four-row assertion, two optimizer-step smoke with finite metrics and both grad norms, then 100 training steps with finite metrics and complete checkpoints at 25/50/75/100. It holds after `.run_done` so fixed-seed checkpoint sampling can be run and artifacts inspected before releasing the GPU node.

- 2026-09-14: Submitted pod `vllm/tdm-overfit-2982ecf-r1`. It initially queued because no full four-GPU node was free, then scheduled intact on `10.0.128.163`; the 4-GPU request was not weakened. It is currently pulling `ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest`.
- 2026-09-14: Prepared but did not yet run `/tmp/issue-775-k8s/tdm-overfit-2982ecf-r1-infer.sh` (SHA-256 `a405cc3561ac1143b9b91e8cddc45f8db0bbdc43ad28452149035f02ac611c16`). After the 100-step marker, it samples the zero-update baseline and checkpoints 25/50/75/100 with the same four-step ladder/seed, generates a 50-step teacher reference, validates MP4 metadata, and records MS-SSIM for inter-rank consistency and change from baseline.

- 2026-09-14: `r1` started on four GB200s and verified exactly four visible GPUs, Torch `2.12.0+cu130`, CUDA 13.0, and exact commit `2982ecf...`. The first pytest import failed before collecting tests: Triton attempted `os.makedirs('/.triton')` as UID 1000 and raised `PermissionError`. No preprocessing, optimizer step, or training ran. Root cause is the launcher omitting the historical `TRITON_CACHE_DIR=/tmp/triton-cache`; implementation code was not implicated. Preparing `r2` with that environment variable.

- 2026-09-14: Corrected `r2` set `TRITON_CACHE_DIR=/tmp/triton-cache`, started immediately on the same cached-image node, and ran the full targeted set. Result: 82 passed, 1 failed in 2.48s. Failure is `test_wan_tdm_validation_propagates_sampling_timesteps_to_dmd_pipeline`: the private-helper test does not initialize `callback.method`, which current-main `ValidationCallback._get_pipeline` now reads for optional scheduler injection. Once supplied, the fake pipeline also needs current `fastvideo_args.pipeline_config` fixture shape. No model preprocessing or training ran because the gate stopped correctly.

- 2026-09-14: Updated only the stale private-helper fixture: it now supplies `callback.method` and the minimal current fake pipeline config shape. Static diff check passed. Committed with a good GPG signature as `aa99f618f` (`[test]: refresh TDM validation fixture`) and immediately pushed to `origin/issue/775-tdm`. No local tests were run.

- 2026-09-14: `r3` ran exact commit `aa99f618ff77f1cb56ccb0fc9c360a50d34b9837` on four GB200s. Targeted tests again reported 82 passed, 1 failed in 2.47s. The prior fixture correction worked and execution progressed; the fake pipeline output then lacked the current callback output contract's `extra` mapping. No preprocessing or training ran.

- 2026-09-14: Added `extra={}` to the validation pipeline stub's output, matching the current callback contract. Static diff check passed. Committed with a good GPG signature as `447ebf2b7412ac694a9fb9048c987104d499e0fc` (`[test]: complete TDM validation pipeline stub`) and immediately pushed. No local tests were run.

- 2026-09-14: `r4` exact commit `447ebf2b7412ac694a9fb9048c987104d499e0fc` passed 83/83 targeted TDM/DMD2/checkpoint tests in 2.43s, found cached Wan snapshot `0fad780a...`, passed four-rank NCCL all-reduce, and produced exactly four parquet shards/four identical prompt rows using all four GPUs. The two-step training smoke passed with finite metrics and both student/critic grad norms; observed step times were about 12-21s. The 100-step run started with the same four-row dataset and JSONL/checkpoint settings.

- 2026-09-14: Live `r4` training reached step 51 with complete checkpoints 25 and 50. Median step time is ~11.9s. The JSONL process only persists global-rank-0 local metrics; with world size 4, SP 1, batch 1, and four trajectory points, transition-aligned sampling assigns indices 0/1/2/3 across ranks each step, while rank 0 consistently reports raw 1000 -> shifted scheduler-label 960. Do not interpret rank-0 loss as an aggregate of all transitions. Rank-0 10-step generator means were 1.71 initially vs 1.88 near step 48, so no local loss decline is established yet; checkpoint media is required.

- 2026-09-14: `r4` completed 100/100 steps at 22:47:30 UTC and wrote `rc=0`, `stage=training_complete`. Checkpoints 25/50/75/100 all contain `.complete`, DCP `.metadata`, and matching step metadata. All 100 rank-0 metric steps are finite. Rank-0 generator loss first/last/median/min/max: 1.8999 / 2.0811 / 1.7867 / 1.3648 / 2.5134; fake-score loss first/last/median: 4.59e-5 / 2.55e-5 / 2.87e-5; student grad norm first/last/median: 1.511 / 1.827 / 1.617; critic grad norm first/last/median: 1.12e-4 / 6.45e-5 / 7.00e-5. Scalar evidence does not show generator-loss overfitting, though critic loss declined; proceed to fixed-seed media gate and do not extend training yet.
- 2026-09-14: Retargeted fixed-seed evaluation script to `r4`; SHA-256 `3961c6fcf50552d06ac1ce9d1574c8105629e430c0f6c39c821def199569e965`. It is syntax-checked and will run only after verified `.run_done`/`rc=0`.

- 2026-09-14: First `r4` evaluation launch failed immediately during import because the separate inference shell did not export `TRITON_CACHE_DIR`; all four ranks attempted unwritable `/.triton`. No video was produced and no training checkpoint was modified. Preserve its runtime directory, add `/tmp/triton-cache`, and relaunch in the same four-GPU pod.

- 2026-09-14: Corrected evaluation export `TRITON_CACHE_DIR=/tmp/triton-cache` and preserved the failed directory as `inference-failed-no-triton-cache`. The retry successfully generated 4 videos each for baseline and checkpoints 25/50/75/100 (20 total), every isolated four-GPU checkpoint process exited cleanly. The optional 50-step teacher then failed before model load because `VideoGenerator` multiprocessing cannot spawn a child from a `<stdin>` main module (`FileNotFoundError: .../repo/<stdin>`). No completed student media was changed. Prepared a continuation using a real `/tmp` Python file plus strict metadata and MS-SSIM evaluation; it will not rerun student inference.

- 2026-09-14: The real-file continuation completed the 50-step teacher and evaluated all 21 videos. Every file is 832x448, 77 frames, 16 FPS, seed 1000. All four ranks are pixel-identical within each baseline/checkpoint group (MS-SSIM 1.0), confirming deterministic replicated inference. Mean MS-SSIM versus baseline is 0.983586 at checkpoint 25, 0.983589 at 50, 0.983570 at 75, and 0.983521 at 100. Successive checkpoint similarity stays ~0.9836. Mean similarity to the teacher is essentially flat: baseline 0.394426; checkpoints 25/50/75/100 are 0.395141/0.394279/0.394283/0.394573.
- 2026-09-14: Visually inspected the contact sheet. Baseline and every trained checkpoint show the same dark, heavily blurred tabletop/interior image with only a vague red blob; there is no recognizable red toy car or visible circular motion. The teacher is sharp and clearly shows a red car on a bright tabletop/studio background. There is no perceptible checkpoint progression toward the prompt or teacher. Combined with the flat-to-higher generator loss, the 100-step overfit gate failed. Per the staged plan, do not spend GPU time on 250/500/1000 steps without first diagnosing the learning path.
- 2026-09-14: Downloaded the evaluation bundle to `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-overfit-447ebf2-r4/`, including all 21 MP4s, contact sheet, comparison JSON, overfit summary, and JSONL metrics. Read-only SHA-256 comparison against Lustre passed for all 23 inference `.mp4`/`.jpg`/`.json` files. Checkpoint integrity check found complete DCP shards at steps 25/50/75/100; every corresponding shard hash differs across checkpoints, so the saved checkpoints are not byte-identical even though their generated media is effectively unchanged.
- 2026-09-14: Final branch verification: clean worktree; local and remote `issue/775-tdm` both point to signed commit `447ebf2b7412ac694a9fb9048c987104d499e0fc`. All four branch commits have good GPG signatures. No PR has been opened and the requested pre-PR review loop remains skipped.
- 2026-09-14: Deleted only Kubernetes pod `vllm/tdm-overfit-447ebf2-r4` after artifact verification, releasing its four GB200 GPUs. PVC/Lustre run state, checkpoints, ConfigMap/manifests, and the verified local artifact copy remain available.
- 2026-09-14: Final GitHub status check as authenticated `macthecadillac` found upstream main advanced after the experiment from `bfc9c017...` to `1c14afd5599a0033d2ffe5db75a4b0d1106b5c58`; remote TDM branch remains `447ebf2b...`, with no open PR. Rebase the four focused commits again onto the new main tip, re-sign/push with an exact lease, and assess overlap before deciding whether any additional four-GPU K8s validation is necessary.
- 2026-09-14: Fetched new upstream main `1c14afd5599a0033d2ffe5db75a4b0d1106b5c58`, read the updated in-scope instructions, and preserved `refs/backup/issue-775-tdm-pre-final-rebase-20260914` at old branch tip `447ebf2b...`. Rebased all four focused commits cleanly with GPG signing. `git range-diff` reports all four patches exactly equivalent; `git diff --check` passes. New branch tip is `bd686f3eb2684f27b6f37f34cb44a64a5dd79b6c`. The two new main commits touch only repository instructions and the forbidden-separate legacy `fastvideo/training/wan_i2v_distillation_pipeline.py` stack plus its test, with no changed-path overlap or dependency on modular `fastvideo/train/` TDM. Additional GPU validation is not proportionate; the exact TDM patches already passed the one-node/four-GPU gate before this no-overlap rebase.
- 2026-09-14: Force-pushed the final rebase with exact lease `447ebf2b...`; GitHub branch now points to `bd686f3eb2684f27b6f37f34cb44a64a5dd79b6c`. Final status is clean and synchronized, `upstream/main...HEAD` is `0 4`, there is no open PR for the branch, and the completed experiment pod is absent. Work segment complete; next useful action is diagnosis rather than a longer copy of the same overfit run.
- 2026-09-15: User approved the deterministic diagnostic plan and requested `/home/sandbox/notify` notification on completion. Re-read the `launch-experiment` skill; continue using this handoff instead of its obsolete experiment-journal location per current repo guidance. Re-checked GitHub as authenticated `macthecadillac`: issue 775 remains open and assigned, its two comments contain no proposed fix, and there is no open TDM/775 PR. The dedicated `/tmp` worktree path was pruned by environment cleanup; local branch `issue/775-tdm` remains durable at `bd686f3e...` and will be reattached at the same path.
- 2026-09-15: Code inspection found a high-priority preflight hypothesis: training LoRA `lora_A` uses Kaiming randomness during role construction, while framework seeding occurs later in `TrainingMethod.on_train_start`; `_replicate_lora_parameters` labels each local tensor as DTensor `Replicate` without an explicit broadcast. Before the 20-50-step probe, run a two-role initialization fingerprint across all four ranks. Rendered syntax-checked `/tmp/tdm-deterministic-init-probe.py` and server-dry-run-valid `/tmp/tdm-deterministic-bd686f3-r1-pod.yaml`. Pod requests exactly four GB200s on one node, reuses the preserved four-row preprocessed data and shared HF cache, checks out exact branch commit `bd686f3e...`, and holds for staged diagnostics.
- 2026-09-15: Submitted `vllm/tdm-deterministic-bd686f3-r1`. Initial manifest unnecessarily reserved 64 CPU/640 GiB and could not schedule; replaced the still-pending pod before execution with the same exact four-GPU request and no artificial CPU/memory reservation. It remains queued because every matching GPU node lacks four currently free GPUs (`36 Insufficient nvidia.com/gpu`, six nonmatching nodes). Do not weaken the one-node/four-GPU constraint. Five long-lived four-GPU pods currently occupy full nodes; keep preparing the probe while waiting.
- 2026-09-15: Prepared full temporary harness `/tmp/tdm-deterministic-update-probe.py` (SHA-256 `8bea6b3b61a58081754ec99d5e73981301e7a37dba949178b64490705d3e3aa1`). It replays each rank's CUDA RNG and identical raw row for 20 steps; records initial/gradient/post-update student and critic LoRA digests/norms at steps 1/5/10/20, globally averaged and per-rank losses, fixed-input generator-latent deltas, transition indices, and final DCP; then a fresh four-rank process reloads the checkpoint and requires exact parameter/output comparison. The harness is temporary diagnostic state, not a new product API or configuration surface.
- 2026-09-15: After ~29 minutes a slot opened and r1 bound intact to node `10.0.128.163` with four GB200s, but setup exited before probe execution (`rc=127`) because the root container PATH lacks `python3`; `/opt/venv/bin/python` is the image's interpreter. No model loaded and no diagnostic/training step ran. Deleted failed r1. A corrected r2 lost the transient slot while re-queuing. Replaced still-pending r2 with `tdm-deterministic-bd686f3-r3`, retaining the exact four-GPU request and using explicit `/opt/venv/bin/{python,torchrun}` paths; r3's setup always falls through to `sleep infinity`, so any future setup issue will preserve the scarce allocation for repair rather than releasing it.
- 2026-09-15: `tdm-deterministic-bd686f3-r3` scheduled intact on node `10.0.135.174` with exactly four visible GB200 GPUs. Setup cloned exact commit `bd686f3e...`; an initial detached-launch preflight used the wrong Lustre mount prefix and exited before `.probe_started`, so no model or step ran. Corrected the launcher to the PVC-root paths `/workspace/issue-775/...` and relaunched. The 20-step deterministic probe is running from `/workspace/run/issue-775/tdm-deterministic-bd686f3-r3`.
- 2026-09-15: The initialization gate proves a distributed LoRA invariant violation before any optimizer step. For both student and critic, all 240 `lora_A` tensors (5,898,240 elements) have four distinct SHA-256 digests across four ranks even though `_replicate_lora_parameters` declares them `DTensor(..., Replicate())`; zero-initialized `lora_B` matches only incidentally. Cause: LoRA A is randomly initialized during role construction before `TrainingMethod.on_train_start` seeds each rank, and `_replicate_lora_parameters` wraps each rank's local value without broadcasting a canonical tensor.
- 2026-09-15: Step 1 and later deterministic milestones independently confirm unsynchronized learning. Identical replayed input/noise/timestep still yields four distinct LoRA gradient digests (all nonzero B gradients; A gradients become nonzero after B updates) and four distinct student/critic parameter digests after the optimizer step. By step 10 all student and critic A/B gradients and parameters differ across ranks. Continue through step 20 and fresh-process DCP reload for a complete before-baseline, then patch initialization synchronization plus LoRA gradient reduction before clipping/optimizer stepping and rerun the same probe on this one-node/four-GPU allocation.
- 2026-09-15: The before-fix run completed 20 steps and saved a complete four-shard DCP checkpoint. Mean generator loss did not improve (1.11107 at step 1 versus 1.11490 at step 20); fake-score loss moved from 1.0784e-4 to 1.0160e-4. At step 20 every student/critic LoRA A/B group still had four distinct parameter digests.
- 2026-09-15: The first fresh-process reload attempt failed only because the temporary probe restored the saved one-row-per-rank dataloader at its exhausted position and then requested a row. Moved the diagnostic row read before checkpoint restore (temporary harness SHA-256 `14dfafe98ec5e5499f083b1806d43e399b6c76ead8922e93254cb3b219051329`) and reran reload without repeating training. Reload then exited 0.
- 2026-09-15: DCP reload exposes the consequence of falsely declared replicas: the reloaded student/critic A/B tensors are each identical across ranks, but the reloaded digests do not match all corresponding in-memory rank states and none of the four fixed generator outputs reload exactly. DCP has necessarily selected/canonicalized one of four divergent local copies, silently discarding the others. This confirms both replica initialization and pre-optimizer gradient synchronization must be fixed before any longer overfit run.
- 2026-09-15: Implemented the candidate fix in the dedicated worktree. `_replicate_lora_parameters` now uses `DTensor.from_local(..., run_check=True)`, whose installed PyTorch contract broadcasts replicated data from the first rank of each mesh dimension. Added a packed `synchronize_lora_gradients` reduction and a standard `TrainingMethod.synchronize_gradients` hook invoked by `Trainer` after all accumulation rounds and before gradient clipping/optimizer stepping. It sums world/SP contributions and divides by `get_dp_world_size()` so distinct data-parallel samples are averaged. Only trainable roles and actual `BaseLayerWithLoRA` gradients participate.
- 2026-09-15: Added focused regression coverage for packed reduction/scaling and trainer-hook dispatch. Uploaded the candidate files to the live four-GPU pod. Targeted pre-commit passed after YAPF formatting; targeted TDM, DMD2, LoRA, trainer, validation, and checkpoint suite passed `143 passed` in 1.50s. No local tests were run. Next: re-check GitHub/main, create a signed focused commit and push it, then rerun the identical 20-step four-GPU probe from the new commit.
- 2026-09-15: Pre-push GitHub check as `macthecadillac` found issue 775 still open/assigned with the same two no-fix comments, no open TDM/775 PR, and unchanged fork branch `bd686f3e...`. Upstream main advanced to `37d06a832f5e8cef3470f81ffb7026e89a405ff0` through five commits. Their changed paths are serving, MiniMax H3, docs, and `fastvideo/dataset/utils.py`; none overlaps the candidate code or the four existing TDM patches. Preserve the patch in a recoverable stash, rebase/re-sign the four TDM commits, push with the exact `bd686f3e...` lease, then restore and commit the fix.
- 2026-09-15: Created safety ref `refs/backup/issue-775-tdm-pre-fix-rebase-20260915`, stashed all candidate changes including the new test, and rebased the four commits cleanly onto `37d06a832...` with `--gpg-sign`. Range-diff shows all four patches exactly equivalent and `git verify-commit` reports good signatures. New rebased tip `9eb51c11360ce8251ac3fc455859a31e84b820ae` was force-pushed with the exact old-tip lease, then the candidate fix was restored without conflict. Worktree now contains only the five intended candidate files.
- 2026-09-15: Committed the fix as `d1e4f676aba57d398f002d938b2edac0b0786ea4` (`[bugfix]: synchronize modular LoRA replicas`), verified its good GPG signature, and immediately pushed it to `origin/issue/775-tdm`. The local worktree is clean. No review loop was run per the user's pre-PR instruction.
- 2026-09-15: Built temporary exact-commit bundle `/tmp/tdm-d1e4f676.bundle` (SHA-256 `112bb0b3a13fadac705fc38384ff8e1fb8bcd6be3984d5b2887c52a35391e3e8`) and fresh pod worktree `repo-fixed-d1e4f676`. Updated only the temporary probe's manual optimizer loop to call the production `method.synchronize_gradients(step)` after backward and before gradient capture/clipping; corrected probe SHA-256 is `e5e64d49926b5b7289a1521f99be233bee5b9c5ff4daee93bd0b636262d22c8f`. Launched a fresh 20-step fixed run plus reload in the same one-node/four-GB200 pod, writing to `probe-output-fixed-d1e4f676`; before-fix output is preserved.
- 2026-09-15: Fixed run passed the initialization and step-1 live gates. Student and critic LoRA A/B each have one digest across all four ranks at construction; after the packed reduction, all step-1 gradient groups have one digest; all post-step parameter groups remain identical. Shared post-reduction grad norms are student 0.317759 and critic 1.02427e-4. The run has continued through step 5 without failure.
- 2026-09-15: Exact-commit fixed probe completed 20 steps plus fresh-process reload with `rc=0`. Every initial, milestone-gradient, post-update, final in-memory, and reloaded student/critic A/B group has exactly one digest across four ranks. All reloaded parameter digests match the in-memory final states and all four rank-local fixed generator tensors reload bit-exactly. Final latent RMS change from the zero-update baseline is nonzero on every rank (0.00577-0.00697). The deterministic mean generator loss remained effectively flat over this short diagnostic (1.11107 to 1.11460), so the correctness gate passes but does not by itself establish overfitting.
- 2026-09-15: Launched the original committed stochastic 100-step overfit recipe at exact commit `d1e4f676a` in the same one-node/four-GB200 pod, with fresh output `overfit-fixed-d1e4f676/output`. Prepared and pod-syntax-checked fixed-seed baseline/checkpoint evaluation plus MS-SSIM/contact-sheet scripts (SHA-256 `41601c5465306547c71d26ccb94d82c20fe23539960f88ee88d650ee4d49e` and `a1783558cc4fde9ee4c75592608d4ec5c1725afb2056195c7856fa0dba3ca67a`). LPIPS is unavailable in the image and FVD is invalid for a single sample, so use registered MS-SSIM, loss/grad trajectories, collapse statistics, and manual contact-sheet inspection.
- 2026-09-15: Post-fix overfit reached step 25 with a complete checkpoint. All metrics remain finite. Rank-0 generator loss is 1.89986 at step 1 and 1.63257 at step 25; first-ten versus latest-ten mean is 1.70552 versus 1.69206. Fake-score loss moved from 4.5855e-5 to 2.6126e-5. Continue to 100, then run the prepared media gate.
- 2026-09-15: The exact-commit post-fix overfit completed 100/100 steps with rc=0; checkpoints 25/50/75/100 are complete. Generator loss did not improve: first/last 1.899855/2.058651, first-ten/last-ten mean 1.705516/1.883702, median 1.791649, min/max 1.367322/2.509441. Fake-score loss decreased modestly: first/last 4.58548e-5/2.65547e-5, first-ten/last-ten mean 3.35342e-5/2.80467e-5. Median step time was 11.919 seconds. The scalar overfit gate fails.
- 2026-09-15: Fixed-seed inference completed with rc=0, producing four rank videos for baseline and checkpoints 25/50/75/100. All within-group rank comparisons are pixel-identical (MS-SSIM 1.0), and the new baseline exactly matches the prior run's baseline, establishing direct comparability. Mean checkpoint similarity to baseline declines monotonically from 0.983548 at step 25 to 0.980113 at step 100. Mean similarity to the 50-step teacher worsens from baseline 0.394426 to 0.394188/0.391190/0.387588/0.381079 at checkpoints 25/50/75/100.
- 2026-09-15: Visually inspected the post-fix contact sheet. Baseline and checkpoints remain a dark, heavily blurred scene with a vague red patch; there is no recognizable toy car or meaningful progression toward the sharp teacher video. Checkpoint 100 is not visibly improved and the quantitative teacher similarity is worse. The post-fix 100-step overfit gate therefore fails. Do not extend training duration; next isolate the TDM generator objective/gradient direction and transition target against a trusted reference or numerical oracle now that distributed replicas and checkpoint reload are proven correct.
- 2026-09-15: Downloaded the final evidence bundle to /home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-fixed-d1e4f676/: before/fixed deterministic reports, fixed probe summary, all 20 fixed-run MP4s, both contact sheets, checkpoint comparison, 100-step summary and tracker records, and relevant logs. Read-only SHA-256 comparison passed for all 49 source/downloaded files with no missing, extra, or mismatched files. Large DCP checkpoints remain on Lustre/PVC and were not duplicated locally.
- 2026-09-15: Final pre-cleanup verification: authenticated GitHub identity is macthecadillac; issue 775 remains open and assigned with unchanged comments; no open TDM/775 PR exists; upstream main remains 37d06a832f5e8cef3470f81ffb7026e89a405ff0; clean local and fork branch tips both equal signed commit d1e4f676aba57d398f002d938b2edac0b0786ea4; branch is five commits ahead and zero behind main. Release only pod vllm/tdm-deterministic-bd686f3-r3 now that artifacts are verified; retain PVC state.
- 2026-09-15: Deleted only Kubernetes pod vllm/tdm-deterministic-bd686f3-r3 after verification, releasing its four GB200 GPUs. PVC run state and checkpoints remain available. Work segment complete; the appropriate next experiment is a minimal objective/gradient-direction oracle, not a longer overfit run.
- 2026-09-15: User requested a rigorous review-only mathematical audit of whether reuse of DMD2 machinery introduced a subtle TDM error. Use the review-code skill, a detached temporary review worktree, original TDM paper/reference implementation, and current runtime evidence. No code or GitHub mutation; report findings by severity and send the requested completion notification.
- 2026-09-15: Audit uses detached worktree /tmp/fastvideo-review-issue-775-tdm-math-audit at exact signed branch tip d1e4f676 and official Luo-Yihong/TDM reference commit 81b019f91539948d49da86cef12ccc64030c5022. Core flow transition, importance-ratio direction, CFG combination, generator-gradient sign, and one-step gradient truncation match the reference derivation; inherited DMD2 loss helpers are not called by TDM.
- 2026-09-15: Preliminary material discrepancies: the failed overfit recipe selects diagnostic-only next_step mode, which sets target sigma equal to the adjacent intermediate boundary and produced beta zero on three of four transitions, rather than sampling tau across each non-overlapping interval as paper TDM does. Also, inherited trainer execution computes both losses before either optimizer step, so the generator observes a stale critic; the official TDM update steps the fake score before computing the generator loss. Continue auditing stochastic defaults, unconditional conditioning, pseudo-Huber semantics, and validation adequacy.
- 2026-09-15: Completed the rigorous review-only mathematical audit. High-confidence conclusions: (1) the failed 100-step overfit is not a faithful TDM validation because noise_interval_mode next_step collapses the sampled target to the adjacent boundary and made transition beta exactly zero for three of four distributed interval assignments; it does not implement the paper random tau inside each non-overlapping interval. (2) Reusing the DMD2 trainer lifecycle creates a material ordering mismatch: FastVideo computes generator and fake-score losses against the same pre-update critic and only then steps both optimizers, whereas official TDM updates the fake score before evaluating/updating the generator. This is most exposed by the diagnostic generator interval of one and needs either a TDM-specific two-phase update or an equivalence experiment.
- 2026-09-15: Additional fidelity findings: the method-level default and docs advertise SDE trajectory sampling even though the TDM paper/reference centers deterministic ODE/DDIM trajectories (both committed Wan recipes explicitly override to ODE, so this did not cause the completed overfit failure); the diagnostic uses all-zero embeddings and an all-zero mask for teacher unconditional CFG rather than an encoded empty/negative prompt; and optional pseudo-Huber follows the compact demo elementwise-plus-DMD-normalization form rather than paper Eq. 11 vector-norm surrogate without DMD normalization. Huber is disabled in the production and diagnostic recipes, so the last discrepancy is latent rather than causal for current evidence.
- 2026-09-15: No core algebraic defect was found in the executed TDM path: TDM overrides the inherited DMD rollout/loss helpers; the flow bridge and mixed-noise identity are correct; importance-weight exponent direction is correct; CFG algebra and real-minus-fake generator target give the intended reverse-KL gradient sign; one-step gradient truncation matches the official method; and no double scheduler shift was found. The principal validation gap is absence of an independent reference/analytic oracle—the current unit tests largely reconstruct expectations with production helpers, and the official public repository contains a PixArt 2D demo but no video-training implementation. Paired same-seed teacher MS-SSIM is also not a primary correctness metric for distribution matching, although the flat loss and visibly blurred samples still establish that the current diagnostic did not learn.
- 2026-09-15: Audit assessment: issue 775 is only partially validated. The fixed-K TDM core is mathematically plausible, but the branch should not be considered algorithmically demonstrated until optimizer ordering is resolved and a faithful separate/ODE/non-zero unconditional experiment plus an independent 1D Gaussian reverse-KL gradient oracle pass. Detached audit worktree is clean at d1e4f676; no tests, code changes, GitHub writes, or GPU jobs were performed during this audit.
- 2026-09-15: User approved the next step. Scope this work segment to the first unresolved gate: add an independent one-dimensional Gaussian reverse-KL oracle for the production TDM generator loss, validate it on Kubernetes using one node and exactly four GPUs, commit with GPG signing, and push immediately if it passes. Do not begin the separate optimizer-ordering production change in the same step, and continue skipping the review loop until PR creation as requested.
- 2026-09-15: Prepared the closed-form oracle patch and a server-dry-run-valid Kubernetes pod. The test invokes the production TDM generator-loss path with exact Gaussian posterior means at sigma 0.5, disables heuristic delta normalization, and asserts gradient equals the analytic derivative of KL(q||p) for generator means below, equal to, and above the real mean. The pod requests exactly four GB200 GPUs on one node and runs the complete local TDM test directory independently once per GPU after targeted pre-commit. Generated asset SHA-256 values: patch 83a75e74..., run script 97516819..., pod manifest 19cf488d....
- 2026-09-15: Oracle pod r1 scheduled immediately on node 10.0.142.67 with exactly four visible GB200s and verified Torch 2.12.0+cu130/CUDA 13.0, then exited before pre-commit/tests because the shared PVC was mounted one directory lower than the prior workload and the exact-commit repo search root was too narrow. No test or source mutation ran. Prepare r2 with a broader read-only source-repo discovery under the mounted PVC; retain r1 status for diagnostics.
- 2026-09-15: Oracle pod r2 also stopped before validation. It mounted the historical PVC root correctly, but a broad find traversed unrelated protected directories because the target fixed checkout is a git worktree whose .git is a file. Narrow inspection located the exact shared parent repository at /workspace/run/run/issue-775/tdm-deterministic-bd686f3-r3/repo; deleted only the still-scanning r2 pod to release its four GPUs. Prepare r3 against that explicit path with no PVC-wide traversal.
- 2026-09-15: Oracle pod r3 found the exact preserved parent repo and verified four GB200s, then stopped before patch/pre-commit/tests because Git rejected the root-owned PVC repository as dubious ownership under pod UID 1000. No validation ran. Prepare r4 with an isolated writable HOME and safe.directory entries scoped only to that exact preserved repository and its git directory.
- 2026-09-15: Oracle pod r4 passed exact source discovery and Git safe-directory handling, then stopped before checkout/tests because local clone attempted object hardlinks across the Lustre PVC and pod-local emptyDir, which returned Invalid cross-device link. No validation ran. Prepare r5 with git clone --local --no-hardlinks; no other behavior changes.
- 2026-09-15: Oracle pod r5 reached the exact local clone but stopped before checkout/tests because the non-root process cannot copy a root-owned promisor pack from the preserved partial clone. No validation ran. Prepare r6 with only the main validation container running as UID/GID 0, matching the owner of the preserved checkout, while keeping allowPrivilegeEscalation false and retaining the exact four-GPU request.
- 2026-09-15: Oracle pod r6 completed successfully on node 10.0.128.163 with exactly four NVIDIA GB200 GPUs visible. Targeted pre-commit completed with the expected test-path exclusions and filename-space check passing. Four isolated processes, one per GPU, each ran the complete tests/local_tests/tdm directory: 41 passed with only 14 existing torch.jit deprecation warnings, approximately 14.35 seconds per process. The new closed-form cases all pass, proving the unnormalized production generator surrogate gradient exactly equals mu_generator - mu_real for unit-variance Gaussians at sigma 0.5. This rules out a core real/fake sign or factor error in that path; optimizer ordering remains the next unresolved discrepancy. Pod phase is Succeeded. GitHub pre-commit check as macthecadillac found issue unchanged, no open TDM PR, upstream main unchanged at 37d06a832, and fork branch unchanged at d1e4f676. Commit and push the oracle test now.
- 2026-09-15: Committed the passing oracle as f1dfbf2a1 ([test]: add TDM Gaussian gradient oracle), verified its good GPG signature, and immediately pushed it to origin/issue/775-tdm. Only tests/local_tests/tdm/test_tdm_method_unit.py changed, adding three closed-form mean cases. Per user instruction, no pre-PR review loop was run. The mathematically appropriate next investigation is now the stale-critic optimizer ordering inherited from DMD2, not further work on generator score sign.
- 2026-09-15: Final oracle-step verification: clean local branch and fork remote both equal signed commit f1dfbf2a1076cc356c70aa74ef9aaaf7a9264ba2; branch is six commits ahead and zero behind upstream main; no tracked handoff files and no open TDM/775 PR exist. Deleted only disposable oracle pods r1-r6 after result persistence; no oracle pods remain. PVC logs/results and the r6 reproducibility ConfigMap remain. This work segment is complete.
- 2026-09-15: User explicitly authorized all future completion notifications through /home/sandbox/notify and states they own the destination server; treat notification-at-completion as a hard rule for the remainder of this thread. User requested both experimental invalidators be fixed: TDM must update the fake-score model before recomputing generator loss, and the diagnostic must replace next_step with faithful separate interval sampling plus real encoded negative-prompt conditioning. Implement as a TDM-owned two-phase managed optimization step, remove the invalid next_step mode, preserve gradient accumulation/synchronization/clipping/schedulers/EMA/checkpoint behavior, validate on one Kubernetes node with exactly four GPUs, then rerun the 100-step overfit and quality gate. Continue skipping review agents until PR creation.

- 2026-09-15 (later, previously unrecorded): Committed the two invalidator fixes as `d7cce9897` (`[bugfix]: restore faithful TDM update order`). `TDMMethod` now overrides `manages_optimization()` and implements a `managed_train_step` that runs critic backward -> critic optimizer step -> then a fresh student trajectory + generator loss against the updated critic. `single_train_step` raises to prevent the default DMD2 ordering. `next_step` mode is removed (`noise_interval_mode` is now only `separate`/`to_terminal`); the overfit recipe switches to `separate`, `cfg_uncond: negative_prompt`/`on_missing: error`, and `max_grad_norm: 1.0` moved into the method. Tests added: `test_tdm_updates_critic_before_generator_recomputes_loss`, `test_tdm_rejects_non_reference_next_step_interval_mode`.

- 2026-09-15: `tdm-faithful-d7cce989` 100-step overfit on one four-GB200 node (rc=0). Confirmed transition_beta now positive (mean 0.653, min 0.061, 100/100 steps) and the negative-prompt encoder loaded. Generator loss still did NOT overfit: first 0.189 -> last-10-mean 0.467 (first-10-mean 0.408), median 0.251. Fake-score loss declined (first 3.35e-5 -> last-10-mean ~1.3e-3... noisy, median 6.9e-5). Quality gate still fails.

- 2026-09-15: `tdm-optimizer-ablation-d7cce989-r1` 50-step sweep: `control` (student 2e-6/critic 8e-6/interval 1), `balanced-high` (8e-6/8e-6), `balanced-low` (2e-6/2e-6), `cadence5` (interval 5), plus `reference-scale` (guidance scale). No variant drove generator loss down; generator loss is dominated by source interval 0 (noisiest, ~0.96 vs ~0.03 for interval 3); cadence5 made fake-score loss rise (second/first ratio 1.66). All variants' teacher MS-SSIM flat ~0.38-0.39. Conclusion: optimizer balance is not the remaining blocker.

- 2026-09-15: `tdm-coupled-trajectory-d7cce989-r1`: candidate patch that reuses the SAME student trajectory for both the critic and generator phases (committed `d7cce9897` regenerates a fresh trajectory per phase). 100 steps, rc=0, 45 pytest passes per GPU. Still failed: teacher MS-SSIM ~0.39 flat (0.3944 baseline -> 0.390 ckpt-100), vs-baseline ~0.981. This directly tests the top finding of the post-d7cce989 audit (see below).

- 2026-09-15: `tdm-shifted-sigma-sweep-d7cce989-r1`: hypothesis that raw timesteps [1000,750,500,250] map (flow_shift 8) to uneven sigmas [1.0,0.96,0.889,0.727] and the last student prediction is made from a very noisy state. Ran inference-only baselines at 4/8/16 steps with evenly-spaced shifted-sigma schedules (`shifted-4/8/16`), plus a `sample_scheduler_point` boundary fix. Final `overfit-8-r2` (8 denoising steps, sigmas [1.0,0.875,...,0.125], 100 steps) still failed: teacher MS-SSIM 0.4303 baseline -> 0.4294 ckpt-100; output pixel_mean 0.36/spatial_std 0.10 vs teacher 0.54/0.386 (dark, blurred).

- 2026-09-15: Post-d7cce989 audit (fresh review agent) produced the definitive finding set. (1) Committed code does NOT share the trajectory between critic and generator; paper Algorithm 1 and the official demo generate one student trajectory per outer iteration and use it for both updates. (2) The paper's principal surrogate (per-sample pseudo-Huber norm, c=0.00054*sqrt(d), no DMD normalization) is not used; recipe uses normalized squared-error, and the current Huber path is elementwise+normalized (matches the demo, not the paper). (3) Production `generator_update_interval: 5` is inherited from DMD2; paper/demo update the generator every iteration. (4) Validation claims `guidance_scale: 6.0` but the Wan DMD stage only does a positive-conditioned forward (effective CFG 1) - misleading but matches the released CogVideoX LoRA example. (5) Wan schedule [1000,750,500,250] -> sigmas [1.0,0.96,0.889,0.727] is weakly justified; effective batch 4 amplifies critic variance vs paper's 32-256. (6) `to_terminal` should stay disabled. Ruled out already: unsynchronized LoRA replicas, zero unconditional conditioning, invalid `next_step`, stale-critic ordering; the independent complete-update oracle passed all four intervals on four GPUs.

- 2026-09-15: Recommended sequence from that audit: (1) fix trajectory coupling first; (2) repeat 100-step overfit; (3) if still failing implement the paper surrogate exactly; (4) raise effective batch to >=32 with identical trajectory replay; (5) ablate timestep ladder + LoRA capacity (full-weight/higher-rank control); (6) improve acceptance criteria (multi-prompt prompt-alignment/motion metrics, not paired student-vs-teacher MS-SSIM). Explicitly: do not spend more compute on 500-1000 steps of the current loop before the coupling test.

- 2026-09-16: Coupling and sigma-sweep experiments have now both been run (100-step) and both still fail, so the two cheapest hypotheses are exhausted. Per user instruction, next run is a 1000-step two-time-scale run (`generator_update_interval: 5`) on the committed `d7cce9897` loop to disambiguate slow two-time-scale convergence from a remaining defect. Remaining un-tested items: paper pseudo-Huber surrogate, effective batch >=32, full-weight/higher-rank LoRA capacity control, and multi-prompt acceptance metrics.

- 2026-09-16: Launched `tdm-1000step-ttur-r2` on one four-GB200 node (10.0.142.67) to disambiguate slow two-time-scale convergence from a remaining defect. Training: 1000 steps, `generator_update_interval: 5` (critic every step, generator every 5th), 4-step denoising [1000,750,500,250] (config default), checkpoints every 250 steps. Code: cloned from public `macthecadillac/FastVideo:issue/775-tdm` @ `b29c868e9` (committed d7cce989 loop, no candidate patch). Smoke (2 steps, interval 1) passed; interval-5 confirmed via metrics (update_student=1 only at steps 5/10/15/...). After training: fixed-seed baseline + checkpoint-{250,500,750,1000} inference, then teacher MS-SSIM + contact-sheet evaluation. Auto-ntfy notification on completion/failure. Branch/handoff committed and pushed at `b29c868e9`.

- 2026-09-16: `tdm-1000step-ttur-r2` completed (rc=0, harness result=pass). Quality result is NEGATIVE. Teacher MS-SSIM monotonically DECLINES with training: baseline 0.3944 -> ckpt-250 0.3937 -> ckpt-500 0.3880 -> ckpt-750 0.3702 -> ckpt-1000 0.3617 (delta -0.0327). The model IS moving: pixel_mean brightens 0.356 -> 0.510 (teacher 0.541), but spatial_std_mean stays ~0.12 (teacher 0.386), so it brightens toward the teacher's mean while remaining a blur. vs-baseline MS-SSIM declines 0.983 -> 0.941. Generator loss noisy/flat (first 0.532, median 0.269, last10-mean 0.370 vs first10 0.292); fake-score loss declines (first10 0.00128 -> last10 0.00026), so the critic tracks but the generator diverges. Conclusion: two-time-scale + 1000 steps rules OUT "too few steps"; this is a genuine objective/optimization problem, signature of reverse-KL mean-seeking/posterior collapse. Artifacts downloaded to outputs/issue-775-tdm/k8s/tdm-1000step-ttur-r2/. Next candidates (in priority order): (1) implement the paper's exact pseudo-Huber surrogate (per-sample vector norm, c=0.00054*sqrt(d), no DMD normalization); (2) directly instrument the real-minus-fake generator gradient direction and critic-vs-teacher divergence; (3) reconsider mean-seeking mitigation (importance weighting / alternative generator target).

- 2026-09-16: Implemented the paper Eq. 11 pseudo-Huber surrogate and ran the 100-step gate. Code committed at `efc9e9ba7` (`[feat]: add paper pseudo-Huber surrogate for TDM generator loss`): new `method.use_pseudo_huber` (bool, default false) computes per-sample vector-norm `sqrt(||pred-target||^2 + c^2) - c` with `c = 0.00054*sqrt(d)` (d = flattened per-sample latent size), skips DMD delta normalization, and is mutually exclusive with `use_huber`. Added `tdm/generator/use_pseudo_huber` metric + 3 unit tests + doc row. Run `tdm-pseudohuber-100step-r1` (4xGB200, interval 1, use_pseudo_huber=true, normalize_generator_delta=false, 100 steps, checkpoints 25/50/75/100). RESULT NEGATIVE: teacher MS-SSIM 0.3944 (baseline) -> 0.3947/0.3940/0.3912/0.3886 (ckpt 25/50/75/100); pixel_mean only 0.356 -> 0.376 (teacher 0.541), spatial_std flat ~0.119 (teacher 0.386). The surrogate is active (use_pseudo_huber=1.0 all steps) but un-normalized loss is large/noisy (gen_loss 15.8-2578, first10 503.7 vs last10 536.3) and the mean-seeking/posterior-collapse signature is unchanged. Conclusion: the loss FORM is not the root cause; reverse-KL mean-seeking persists. Artifacts in outputs/issue-775-tdm/k8s/tdm-pseudohuber-100step-r1/. Remaining candidates: instrument real-minus-fake generator gradient direction + critic-vs-teacher divergence directly; effective batch >=32; full-weight/higher-rank LoRA capacity; importance weighting / alternative generator target (reverse-KL -> forward-KL).

- 2026-09-16: User clarified that the previous experiments establish only three separate failures: more training does not rescue uncoupled normalized L2, pseudo-Huber does not rescue the uncoupled loop, and coupling alone does not rescue normalized L2. They do not rule out the faithful interaction of same-trajectory coupling with the paper surrogate. First commit the previously validated same-trajectory reuse and strict interval lower bound atop `efc9e9ba7`, then run the clean 100-step combined control on one node/four GPUs. Continue skipping review agents until PR creation and notify through `/home/sandbox/notify` at completion.

- 2026-09-16: Implemented the narrow candidate atop `b9fc08157`: each accumulated raw batch is prepared once, its detached student trajectory is retained through the critic optimizer step, and the generator phase resamples an independent context from that same trajectory. `_tdm_fake_score_loss` now receives the trajectory explicitly. Scheduler candidate selection now requires `scheduler_sigma >= lower` rather than admitting values down to `lower - 1e-6`; the strict transition invariant remains unchanged. Added a combined coupling+pseudo-Huber regression, the deterministic one-ULP shifted-scheduler boundary regression, updated direct fake-loss unit calls, and corrected the durable trainer documentation. `git diff --check` passes; no local tests were run. Candidate patch SHA-256 is `bfcfec53f91e27768e4cb4e6758a25b5aaf1f410a84060936cf410495d31e841`; exact-four-GB200 held-pod manifest SHA-256 is `035c13257abb6a6415c6dce4896f8b496c542e0fcf5dc2876c86ce32664b94e9` and passed server dry-run. Validate pre-commit, the complete local TDM tests independently on every GPU, and a two-step combined production smoke before signing/committing/pushing only the code, tests, and docs.

- 2026-09-16: Candidate validation passed on pod `vllm/tdm-coupled-pseudohuber-r1`, node `10.0.142.67`, with request=limit=4 GB200 GPUs and exactly four visible devices. Targeted pre-commit passed every applicable hook. Four independent processes, one per GPU, each ran the complete `tests/local_tests/tdm` suite and reported `45 passed, 14 warnings`. A four-rank two-step production smoke with `generator_update_interval=1`, `use_pseudo_huber=true`, and `normalize_generator_delta=false` completed with finite metrics and the intended flags on both rows; generator losses were `750.29` and `110.29`. Validated pod/local files are byte-identical. Re-check GitHub/main, create one focused GPG-signed commit excluding this handoff, push immediately, then run the exact committed 100-step combined control in the held pod.

- 2026-09-16: Re-checked GitHub as `macthecadillac`: issue 775 remains open/assigned with two unchanged comments, no TDM/775 PR exists, upstream main remains `9b0e57fe...`, and fork tip was `b9fc08157`. Committed only the validated code/tests/docs as `6676ef6b1` (`[bugfix]: couple TDM trajectory updates`), verified its good GPG signature, and immediately pushed it to `origin/issue/775-tdm`. The active handoff remains a local unstaged modification. Run the combined 100-step gate from exact commit `6676ef6b1` in the held four-GB200 pod; do not use the patch-applied validation checkout as experimental provenance.

- 2026-09-16: Prepared the exact-commit `tdm-coupled-pseudohuber-100step-r1` control. It clones and checks out `6676ef6b10ae240f0cb62dc8bf3d959faf777819`, runs a fresh two-step smoke, then 100 updates with the same four-step overfit recipe, seed/data, LRs, `generator_update_interval=1`, `use_pseudo_huber=true`, and `normalize_generator_delta=false`, retaining checkpoints 25/50/75/100. It validates finite metrics/config/checkpoint integrity, samples baseline plus every checkpoint on all four GPUs, and produces the same teacher MS-SSIM/media-stat/contact-sheet report as the prior uncoupled pseudo-Huber gate. Syntax-checked launcher SHA-256 is `5799ba1cf0f90295fe4406b07e34d6cb5b1cf100d572a5e9ba59f258a183563f`; unchanged evaluator SHA-256 is `eaa5380ec066bd2dc027fda67cffe52caea3745d9f985794d76c44f0b1668923`; pod uploads match exactly.

- 2026-09-16: The exact-commit combined control completed end-to-end with `rc=0`: fresh four-rank smoke, 100 finite updates, complete checkpoints 25/50/75/100, 20 deterministic videos, and evaluation/contact sheet. Scalar behavior failed: generator-loss first/second-half means were `304.58`/`410.28` (ratio `1.347`), fake-score means were `0.000565`/`0.000715` (ratio `1.265`), median student/critic grad norms were `151.66`/`0.000174`, and median step time was `9.459s`.

- 2026-09-16: The media gate also failed. Teacher MS-SSIM was baseline `0.394426`, checkpoint 25 `0.394497`, checkpoint 50 `0.393869`, checkpoint 75 `0.392273`, and checkpoint 100 `0.390995`; similarity to baseline declined to `0.981836` at step 100. Pixel mean moved `0.3563 -> 0.3729`, but spatial standard deviation stayed essentially flat `0.1176 -> 0.1190` versus teacher `0.3861`. Visual inspection shows every checkpoint remains the same blurred interior/tabletop with an indistinct red object and no progression toward the sharp red-car teacher. Compared with uncoupled pseudo-Huber, coupled step-100 teacher MS-SSIM is slightly less degraded (`0.390995` versus `0.388570`), but the difference is not a meaningful learning result; direct coupled-versus-uncoupled checkpoint MS-SSIM remains about `0.9836`. Therefore faithful coupling plus the paper surrogate does not rescue this 100-step gate. Persisted direct comparison SHA-256 is `8c52e402caf77184cf898c5415c652dcaefd98ea24d4bd3e1bbb82b423a3298e`. Next highest-value work is direct instrumentation of critic-versus-teacher error and the resulting generator gradient by interval, not another recipe-only overfit run.

- 2026-09-16: Preserved a checksum-verified compact evidence archive at `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-coupled-pseudohuber-100step-r1/`. Archive SHA-256 is `5e01545ea311b7a07ce460937b2525bdb6fc40689895fb26049bf85b26e64b1b`; the extracted evidence contains all 20 MP4s, contact sheet, comparison JSONs, 100-step metrics/config/summary, logs, candidate validation and smoke records, and exact launcher/evaluator/manifest/patch inputs. Key extracted hashes match the PVC source. Complete DCP checkpoints remain on the PVC and were excluded from the local archive.

- 2026-09-16: Verified no training/inference process remained and deleted only pod `vllm/tdm-coupled-pseudohuber-r1`, releasing its four GB200 GPUs. PVC run state/checkpoints, ConfigMap assets, and the verified local evidence remain. Final GitHub check as `macthecadillac`: issue 775 remains open and assigned with two unchanged comments, no TDM/775 PR exists, upstream main remains `9b0e57fe...`, and local/fork tips both equal signed commit `6676ef6b1`. The only local modification is this active handoff; `git diff --check` passes.

- 2026-09-16: User approved the next diagnostic. Re-checked GitHub as `macthecadillac`: issue 775 is still open/assigned with the same two comments, no open PR targets TDM/775, upstream main is still `9b0e57fe...`, and the local/fork branch remains `6676ef6b1`. The diagnostic will use the exact production loss on independently sampled but fixed same-trajectory contexts for each of the four intervals. For both initialization and the failed coupled+pseudo-Huber checkpoint 100, reset the critic to an identical phase snapshot before each interval; measure supervised critic error on train and held-out contexts before/after one real four-rank critic update; and compare the synchronized student LoRA gradient before/after that update against a sample-target-oracle gradient. This isolates critic learning/generalization, gradient direction, and interval dependence without another blind training run. Run only on one Kubernetes node with exactly four GPUs; do not commit a temporary probe harness or begin the pre-PR review loop.

- 2026-09-16: Rendered temporary job `tdm-critic-gradient-6676ef6-r1`. It checks out exact commit `6676ef6b10ae240f0cb62dc8bf3d959faf777819`, requires the retained combined-run checkpoint 100, and evaluates initialization plus checkpoint 100 across all four intervals. Every interval starts from the same phase-specific critic parameters/optimizer/scheduler; four ranks contribute independent fixed train/eval contexts from one underlying trajectory; the critic takes one production update; and the probe compares production pseudo-Huber student gradients before/after against a sample-target oracle. Python AST, shell syntax, and Kubernetes server dry-run passed. SHA-256: harness `67d347a3d12201490f7cbd3a9616edd2f73ff11376f5722f853987fcc3b7cc5f`, runner `58407cb631bd094ba42f9e5f90a8cdb50bf37ec56cd4ceb0d6f5f766923b1156`, manifest `82bd295ee8993ab027bc1ce733119cba9f273ce991d7af4b4566679e83be0468`. Submit one pod requesting exactly four GB200 GPUs; preserve JSON/log evidence before releasing it.

- 2026-09-16: `tdm-critic-gradient-6676ef6-r1` completed successfully on node `10.0.142.67` with exactly four GB200 GPUs, exact commit `6676ef6b1`, both phases, all four intervals, and finite metrics. At initialization the critic exactly equals the conditional teacher (`fake_to_teacher_cond_rms=0`), as expected from identical base weights and zero-initialized LoRA output. At checkpoint 100 it is still much closer to the conditional teacher than to generated samples: per-interval fake-to-teacher RMS is only 8.5%, 16.8%, 26.9%, and 40.8% of fake-to-student RMS.

- 2026-09-16: A real four-rank critic update barely improves its own task. At checkpoint 100, training-context RMS ratios are `0.99871/0.99890/0.99919/0.99952` and held-out RMS ratios are `0.99981/1.00067/0.99866/0.99971` for intervals 0-3. The held-out prediction change has only `0.016/0.015/0.066/0.044` cosine with the direction toward the clean generated target; individual ranks include negative cosines. Correspondingly, pre/post synchronized student-gradient cosines are `0.999996/0.999994/0.999980/0.999344`: the supposedly fresh critic update has practically no effect on the generator direction.

- 2026-09-16: The checkpoint-100 generator gradient itself is interval-imbalanced and becomes invalid on the last interval relative to the fixed-context single-sample target oracle. Pre-update gradient cosines to that oracle are `0.695/0.956/0.743/-0.310`, while gradient norms are `868.5/266.0/44.3/8.58` (about 101x first-to-last). Latent real-minus-fake cosines remain positive (`0.945/0.927/0.921/0.733`), showing that model-Jacobian projection turns the final interval's imperfect critic delta into an oppositely directed parameter update. The oracle is a supervised single-sample diagnostic, not the unknown exact conditional expectation, so the negative final-interval result is evidence of critic-induced distortion rather than a proof that the mathematical TDM target is wrong.

- 2026-09-16: Critic and generator update allocation are opposed. On checkpoint-100 sampled contexts, the critic SNR component averages approximately `0.000421/0.00590/0.0303/1.55` across intervals, a roughly 3,700x last-to-first ratio, and critic gradient norms rise from `5.43e-5` to `2.59e-4`. The student gradient instead falls by about 101x from first to last. This, plus the near-orthogonal held-out critic update, points first to critic variance/interval weighting rather than the already-oracled core generator algebra. The cheapest next discriminator is the same critic-only probe at effective batch 32; only if alignment materially improves should a full batch-32 overfit run follow.

- 2026-09-16: Full evidence is at `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-critic-gradient-6676ef6-r1/`. Local/PVC hashes match: `diagnostic.json` SHA-256 `adef199c26bcec59e90855b2938fe8b52a3331276d2eb57701acb8f3cec0b155`; `summary.json` SHA-256 `c2af72d0609958d98aa238dc5d274839dbbf3dd17c641af1930d9e3a60e71bc4`. The complete PVC run and ConfigMap remain. Deleted only the completed diagnostic pod and temporary artifact-access pod, releasing all four GPUs. No production code changed and no commit/PR/review loop was needed.

- 2026-09-16: User authorized the effective-critic-batch-32 discriminator. Refreshed GitHub as authenticated `macthecadillac`: issue 775 remains open/assigned with the same two comments, no open PR targets TDM/775, upstream main remains `9b0e57fe...`, and local/fork branch tips both remain signed commit `6676ef6b1`. The temporary probe will compare batch 4 and batch 32 directly from identical critic/optimizer/scheduler snapshots in one job: one versus eight independent training trajectories per rank, the batch-4 trajectory as a subset of batch 32, and one separate fixed held-out trajectory/context shared by both arms. The student stays frozen; each arm takes exactly one production critic update and is evaluated by held-out critic-target error/update alignment and the resulting production student gradient versus the same sample-target oracle. Run on one Kubernetes node with exactly four GB200 GPUs. Proceed to a full 100-update batch-32 coupled+pseudo-Huber control only if this paired gate shows materially better held-out alignment; otherwise stop compute and investigate interval weighting/update allocation.

- 2026-09-16: Prepared temporary job `tdm-critic-batch32-6676ef6-r1`. The harness exactly matches production accumulation semantics: each local critic loss is divided by 1 or 8 in `WanModel.backward`, then replicated LoRA gradients are summed across all four ranks and divided by the data-parallel world size before the one optimizer step. It covers initialization and retained combined-control checkpoint 100, resets all critic/optimizer/scheduler state between arms and intervals, and preserves a common held-out trajectory per comparison. Python AST and shell syntax checks passed; the pod manifest passed Kubernetes server dry-run and requests/limits exactly four GB200 GPUs on one node. No existing issue-775 pod conflicts. Input SHA-256: harness `5b7cd6d051f5f83f5a1a9bf37944e935cd270e8760cdfa7f312e8ba9cb3dc14f`, runner `5373d1e14dae813e788f031d1db00c25ddcf55e34e5f5e56c6bccc4d549f87d3`, manifest `bd2d446157bfc653c66f91871c2d7f81067b98f41a4eef4f36eff3e4c066d106`. Submit the ConfigMap/pod, monitor to completion, copy and checksum JSON/log evidence while the pod is held, then delete only the pod to release all GPUs.

- 2026-09-16: Created ConfigMap `tdm-critic-batch32-6676ef6-r1-assets` and submitted pod `vllm/tdm-critic-batch32-6676ef6-r1`. The scheduler has kept it Pending because no matching node currently has four GPUs free (`36 Insufficient nvidia.com/gpu`, six nodes do not match the GB200 selector). The pod is valid and requests/limits remain exactly four GPUs; do not weaken or split the one-node/four-GPU constraint. Continue monitoring until capacity opens.

- 2026-09-16: After a 48-minute capacity wait, `tdm-critic-batch32-6676ef6-r1` scheduled intact on node `10.0.129.200`. The node pulled the development image, exposed exactly four NVIDIA GB200s with 189471 MiB each, reported Torch `2.12.0+cu130`/CUDA 13.0, cloned and detached exact commit `6676ef6b1`, passed the in-container diagnostic syntax gate, and began the four-rank probe at `2026-09-16T18:16:57Z`. Monitor logs/result markers; do not launch the full training control until the paired checkpoint-100 gate is analyzed.

- 2026-09-16: `tdm-critic-batch32-6676ef6-r1` completed successfully at `18:26:25Z` with both phases, four intervals, paired batch-4/batch-32 arms, exact shared held-out contexts, finite results, and `rc=0`. The batch-size gate is NEGATIVE. At checkpoint 100, batch 32 improved held-out RMS in only 2/4 intervals and held-out update cosine in only 2/4. Mean batch32-minus-batch4 held-out RMS ratio was `-0.000347` (only 0.035% mean improvement), mean update-cosine change was `+0.00392`, and mean generator-gradient cosine-to-oracle change was `+0.000954`. Per-interval held-out update cosine moved from batch 4 `0.0284/0.0271/0.0616/0.0393` to batch 32 `0.0200/0.0217/0.0641/0.0664`; the first two worsened. Post-update batch4-versus-batch32 student-gradient cosines were `0.999996/0.999995/0.999969/0.999194`, so the two critic updates induce effectively the same generator direction.

- 2026-09-16: The previously problematic last interval remains invalid relative to the fixed sample-target oracle: pre-update generator-gradient cosine is `-0.30964`; post-update it is `-0.30809` at batch 4 and `-0.30703` at batch 32. Batch 32 does make the last interval's held-out critic RMS ratio `0.99793` instead of `1.00012`, but that larger movement is still almost orthogonal to the supervised target (cosine `0.0664`) and does not repair the negative generator gradient. In the first three intervals the accumulated critic gradient norm is smaller than batch 4, consistent with cancellation of noisy per-sample gradients, without a compensating alignment gain. Therefore effective batch 32 alone does not address the defect; do not run the 100-update batch-32 coupled+pseudo-Huber control. The next investigation should isolate critic interval weighting/update allocation (especially the approximately 3,700x SNR weighting opposition already measured), rather than spend on another recipe-only run.

- 2026-09-16: Preserved the complete 384 KiB evidence directory at `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-critic-batch32-6676ef6-r1/`; local and PVC hashes match. SHA-256: `diagnostic.json` `3b74da501b8e64cbb2c0634978e0691decfbc82f9553d3589efff23a03e084aa`, `summary.json` `1d1a492a73b6aa414ea20ee36b3989124373e9f9ba2ef97903fe5e779311b8ad`, diagnostic log `3e2fb47a16b74d9ab793f1df0feaf4814f33f9a21fd814724d78c25a52a8966d`, orchestrator log `31dc1262c60dbbcf16572b4f311e8bfcf3fa5cde5b06d420bdbe04875ffe624f`. Delete only the held completed pod to release the four GPUs; retain PVC evidence and ConfigMap provenance. No production code changed, so no commit/review loop is required.

- 2026-09-16: Deleted only completed pod `vllm/tdm-critic-batch32-6676ef6-r1`; `kubectl get pods -l issue=775` now reports no resources, so all four GPUs are released. Retained the PVC run and ConfigMap. Final GitHub check as `macthecadillac`: issue 775 is open/assigned with unchanged comments, no PR exists for `macthecadillac:issue/775-tdm`, upstream main remains `9b0e57fe...`, and local/fork tips remain signed commit `6676ef6b1`. The sole worktree modification is this mandatory local handoff. Work segment complete; next action is the mixed-interval critic-gradient decomposition described at the top of this handoff, not a full batch-32 training run.

- 2026-09-16: User authorized the mixed-interval critic-gradient decomposition and conditional weighting control. Re-read the `launch-experiment` skill; its experiment-journal instruction remains superseded by current repository policy, so all state stays in this mandatory handoff. Refreshed GitHub as `macthecadillac`: issue 775 remains open/assigned with two unchanged comments, no PR exists for the TDM branch, upstream main remains `9b0e57fe...`, and local/fork tips remain signed commit `6676ef6b1`. First decompose the exact production mixed-interval critic gradient at checkpoint 100 into per-interval LoRA vectors, norms, pairwise cosines, and clipped-SNR/importance contributions. Only if that shows a material allocation conflict should the same fixed trajectories/contexts be replayed through production versus interval-normalized/importance-only weighting controls. Use one Kubernetes node with exactly four GB200 GPUs; keep the student frozen; do not modify production code or start the review loop.

- 2026-09-16: Prepared temporary job `tdm-mixed-weight-6676ef6-r1`. Four deterministic replicas each map rank `r` to interval `r`, yielding the production-shaped mixed global batch while retaining interval attribution. For each replica the probe captures unsynchronized per-rank critic LoRA gradients under production, mean-matched importance-only, and mean-matched unit weights; all-gathers the vectors; and reports norms, projections onto the aggregate, pairwise cosines, aggregate directions, exact SNR/importance weights, and the corresponding per-interval student gradients. The conditional gate is predeclared as median SNR last/first >=100, production critic-gradient last/first >=3, and student-gradient first/last >=10 across the four replicas. If all hold, the exact first-replica training contexts are replayed from identical checkpoint-100 parameters/optimizer/scheduler through one synchronized production, importance-only, and unit-weight critic update and evaluated on separate held-out trajectories; otherwise no update control runs. All control weights are globally mean-matched to production so the comparison isolates allocation rather than overall loss scale. Student parameters are never stepped.

- 2026-09-16: Python AST, shell syntax, and Kubernetes server dry-run passed; the manifest requests/limits exactly four GB200 GPUs on one node and no issue-775 pod conflicts exist. SHA-256: harness `5a5898bcc439971d684bc301d4768202ed3a8161fade057253b0c4863d355d78`, runner `00976c5477420e7525415f0f57dd0062d2c8d24929a2abd5c9eda5a49e431277`, manifest `8dc656f6639f5cb1d799c5f5e97064492b26e40db124760f173c391c92874a04`. Submit the ConfigMap/pod, monitor the gate and conditional controls, preserve/checksum JSON and logs, then release the GPUs. Do not start a full training run in this job.
### 2026-09-16 mixed-interval weighting diagnostic submitted

- Submitted Kubernetes pod `tdm-mixed-weight-6676ef6-r1` in namespace `vllm` from `/tmp/issue-775-k8s/tdm-mixed-weight-6676ef6-r1-pod.yaml`.
- The pod scheduled immediately on node `10.0.129.200` with exactly four visible NVIDIA GB200 GPUs (189471 MiB each), satisfying the one-node/four-GPU constraint.
- Startup log verified exact branch tip `6676ef6` and launched the diagnostic at `2026-09-16T18:45:14Z`.
- Current state: running; no failure has been observed. Monitor `diagnostic.log` and the pod completion marker, then copy and checksum the retained evidence before deleting the completed pod.

### 2026-09-16 mixed-interval diagnostic r1 harness failure

- `tdm-mixed-weight-6676ef6-r1` failed at `2026-09-16T18:46:35Z` with `rc=1` before producing a diagnostic report. This was a temporary harness bug, not a TDM result.
- Retrieved the retained traceback from the shared volume using short-lived four-GPU pod `tdm-mixed-weight-6676ef6-r1-inspect`. All four ranks failed identically in `decomposition_round`: `capture_local_generator_gradient` returns `(metrics_dict, gradient_tensor)`, while the decomposition call assigned the results in reverse order and passed the dict to `torch.empty_like`.
- Minimal fix: reverse the two local assignments at the decomposition call site. No production code or experimental definition changed.
- Prepared retry identity `tdm-mixed-weight-6676ef6-r2`. Its runner now prints the retained diagnostic tail into the pod log on failure so another retrieval pod will not be needed.

### 2026-09-16 mixed-interval diagnostic r2 submitted

- Released failed pod `tdm-mixed-weight-6676ef6-r1` and succeeded retrieval pod `tdm-mixed-weight-6676ef6-r1-inspect` after verifying the retained traceback. The shared-volume r1 failure evidence remains intact.
- Corrected r2 assets passed Python AST parsing, `bash -n`, and Kubernetes server-side dry run. SHA256:
  - diagnostic: `a4fabba463ad7490020b03f5369b806d0e32e0033e2c376e2b01cf0580222b66`
  - runner: `0919dfc8b1876732c0e7b03c93cd1eed2281023ce7319725de9414c8f82e5802`
  - manifest: `a15893bab612008529c059cbb140930a9357fc0fd105b5f1a492bf489676b265`
- Created immutable ConfigMap `tdm-mixed-weight-6676ef6-r2-assets` and submitted pod `tdm-mixed-weight-6676ef6-r2` in namespace `vllm`.
- The pod scheduled immediately on node `10.0.129.200` and is running. Continue monitoring; do not interpret r1 as an experimental result.

### 2026-09-16 mixed-interval diagnostic r2 completed

- `tdm-mixed-weight-6676ef6-r2` completed successfully at `2026-09-16T18:55:06Z` on one node with exactly four GB200 GPUs, exact commit `6676ef6b1`, four deterministic mixed-interval replicas, the predeclared gate, and all three conditional one-step controls. Student parameters were never stepped.
- The predeclared allocation gate passed all three conditions in every replica. Across replicas, the median last/first clipped-SNR ratio is `32691.35x`, the median production critic-gradient last/first norm ratio is `40.85x` (per-replica `58.22/3.61/72.90/23.47x`), and the median student-gradient first/last norm ratio is `173.38x` (per-replica `210.32/67.73/170.65/176.11x`). The last interval supplies a median `94.56%` of the synchronized production critic update projection. Thus the production critic update is overwhelmingly allocated to the interval where the student parameter gradient is smallest.
- Removing SNR weighting changes the critic direction almost completely: median aggregate production-to-importance-only cosine is `0.000055`; importance-only and unit directions have cosine above `0.999996`, confirming the importance term itself is approximately one and clipped SNR causes the direction change.
- The conditional controls do **not** show that removing SNR weighting is a fix. On the exact first-replica contexts, production improved held-out critic RMS in 3/4 intervals and had mean held-out RMS ratio `0.999189` with mean update-to-target cosine `+0.04215`. Importance-only and unit each improved only 1/4 intervals, with mean held-out RMS ratios `1.006113`/`1.006283` and mean update cosines `-0.03954`/`-0.04308`. Mean post-update generator-gradient cosine to the sample-target oracle was also slightly higher for production (`0.54997`) than importance-only (`0.54570`) or unit (`0.54624`). The last interval remained oppositely directed for all arms (`-0.6603/-0.6672/-0.6652`).
- Important control limitation: global mean-matching of loss weights did not match the effective AdamW step. Importance-only/unit produced parameter deltas `14.374x`/`14.377x` larger than production (`0.17203/0.17207` versus `0.01197` L2), and strongly fit interval-0 training context while worsening its held-out error by `1.80-1.87%`. Therefore this result rules out a naive same-LR removal of SNR weighting, but it does not isolate whether an alternative update direction is better at the production step size.
- Preserved the complete 192 KiB evidence at `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-mixed-weight-6676ef6-r2/`; local and PVC hashes match. SHA-256: `diagnostic.json` `eb3d231b8bb9afda1d6fa14184cfb1d2c588d35d0ee7be1497aab836d5ffcf55`, `summary.json` `8dc768bddd601714096eba03557b39733172ddf87f83fabda488a67fd5acc3d8`, diagnostic log `49d1e822c414e55b9011da3e8d00480e91cc3b84fbda1f25995399c914dc59a8`, orchestrator log `da9e239c0556d536b882d4b6376bb892b2e42cd7d6e6bb4ee9c7207b98217ac6`.
- Deleted the completed r2 pod after checksum verification, releasing all four GPUs; the PVC run and immutable ConfigMap remain. Final checks show no issue-775 pods, no TDM PR, issue 775 still open/assigned with unchanged comments, upstream main `9b0e57fe...`, and local/fork branch tips both at signed commit `6676ef6b1`. The only worktree modification is this mandatory handoff.
- Recommended next discriminator: a one-step direction-only control. Generate each arm's exact AdamW parameter delta from the same snapshot, rescale alternative deltas to the production delta norm before applying them, repeat across all four deterministic replicas, and evaluate multiple held-out trajectories per interval. Do not launch a 100-update alternative-weighting run unless that norm-matched gate is positive.

### 2026-09-16 critic-capacity hypothesis and proposed gate

- User supplied an independent assessment ranking critic LoRA capacity/target-module coverage above interval weighting, student LoRA capacity, and VSA. The evidence supports critic capacity as a genuinely high-value next hypothesis: LoRA synchronization/update/checkpoint mechanics are verified, the checkpoint-100 critic remains teacher-like, the final interval generator gradient is still badly misdirected, and batch 32 did not materially improve it.
- Important qualifications: the negative batch-size result rules out variance reduction as a sufficient one-step fix but does not itself prove a capacity limit; rank and target-module coverage are distinct; and the same-student-LoRA oracle makes the critic the proximate source of the actual-versus-oracle discrepancy, while the student Jacobian can still amplify critic error. The sample-target oracle is diagnostic rather than the exact conditional-expectation target.
- A clean capacity ceiling should remove interval weighting as a confound first. Freeze the student and reuse identical generated trajectories, train/evaluate intervals independently or with a balanced mixed design, and compare current rank-16/current targets against a substantially larger rank on the same targets and a full-weight critic. Record train and multiple-heldout curves, critic update-to-target alignment, and resulting generator-gradient alignment per interval. Track effective parameter/output update norms so AdamW step-size differences are not mistaken for capacity.
- Minimal first comparison should be rank 16, rank 128, and full weight; add rank 64 only if rank 128 materially helps and a scaling curve is needed. A positive capacity gate should require a material held-out improvement, including the last interval, plus a substantial repair of the last-interval generator-gradient cosine—not merely faster training-context fitting. Only then test the winning critic capacity in coupled TDM with the student LoRA unchanged.
- Current judgment: critic capacity is a sound next experiment, but it is not yet established as the leading root cause over interval weighting/optimizer dynamics. Prefer the critic-capacity ceiling before a full 100-update alternative-weighting run; retain the norm-matched weighting control as the other unresolved discriminator.

### 2026-09-16 concrete critic-capacity experiment plan (not launched)

- Goal: determine whether the critic's current attention-only rank-16 LoRA parameterization prevents it from learning the checkpoint-100 student's generated distribution and producing a useful generator gradient. Do not update the student and do not launch coupled TDM in this gate.
- Common start/data: freeze the retained checkpoint-100 student; generate one deterministic bank shared by every arm, with 16 fixed training and 16 disjoint held-out trajectories/contexts per interval. Start every critic arm from the same pretrained teacher function with fresh optimizer state, rather than giving the existing rank-16 checkpoint an initialization advantage. Rank `r` remains fixed to interval `r`, but use balanced/unit critic weights in this ceiling test so the already-demonstrated SNR allocation does not confound capacity.
- Initial arms, run sequentially in one one-node/four-GB200 pod: (A) production target modules `to_q/to_k/to_v/to_out`, rank 16, alpha 32; (B) the same target modules, rank 128, alpha 256, preserving alpha/r=2; (C) full-weight critic. Omit rank 64 initially; add it only if rank 128 helps and a scaling curve is needed.
- Optimization: 50 critic-only updates with checkpoints/evaluation at 0/1/10/25/50; AdamW betas `[0.0, 0.999]`, weight decay `0.01`, max grad norm `1.0`, fresh state. Use rank-16 at `8e-6` as the reference and calibrate rank-128/full-weight learning rates using only first-step critic-output RMS so the first functional update magnitude matches rank 16. Log parameter-delta and output-delta norms throughout to expose optimizer/parameterization drift.
- Evaluation: multiple held-out contexts per interval, reporting training and held-out critic-to-clean-target RMS, held-out update-to-target cosine, and median resulting student generator-gradient cosine to the fixed-context sample-target oracle. The sample oracle remains a diagnostic, not the exact conditional expectation.
- Positive gate: a higher-capacity arm must beat rank 16 by at least 10% held-out critic RMS in at least 3/4 intervals including interval 3; interval-3 median generator-gradient cosine must improve by at least 0.2 and become positive; and the gain must not be training-only memorization. A merely faster training-loss decline is insufficient.
- Interpretation: rank128 approximately full > rank16 indicates rank limitation; full > rank128 indicates missing target-module coverage or a remaining low-rank constraint and warrants one expanded-target LoRA follow-up; all arms similar means capacity is unlikely and the next discriminator is the norm-matched weighting/update-direction control; held-out improvement without generator-gradient repair means capacity alone is not sufficient for coupled training.
- Only after a positive gate: reproduce the winning capacity from the actual checkpoint-100 critic function by merging the existing critic adapter into the base (with an exact pre/post function-equality assertion), then run a short continuation. Launch a full coupled 100-update TDM run only if that continuation also improves held-out critic and generator-gradient alignment.
- Operational constraints: temporary harness only, exact signed branch commit `6676ef6b1`, one Kubernetes node with exactly four GB200 GPUs, sequential arms to control memory, checksum-verified JSON/log evidence, and release the pod after copying artifacts. No production code, commit, PR, or review loop is planned for this diagnostic. Expected runtime is roughly 45-90 minutes after scheduling. The experiment has not been submitted.

### 2026-09-16 critic-capacity gate authorized

- User authorized execution of the concrete critic-capacity plan. Re-read the `launch-experiment` skill; its legacy experiment-journal instruction remains superseded by repository policy, so active state stays in this mandatory handoff.
- Refreshed GitHub as authenticated `macthecadillac`: issue 775 remains open and assigned with the same two comments and no proposed fix; no open TDM PR and no PR for `macthecadillac:issue/775-tdm` exists. Refreshed refs: local and fork branch tips both remain exact signed commit `6676ef6b10ae240f0cb62dc8bf3d959faf777819`; upstream main remains `9b0e57fe4b3a8c112ff24cc22f752eb0608ccfab`.
- No issue-775 Kubernetes pod is active. Proceed to build a temporary capacity harness and one-node/four-GB200 manifest. Do not change production code or begin the pre-PR review loop.

### 2026-09-16 critic-capacity gate prepared

- Prepared temporary job `tdm-critic-capacity-6676ef6-r1`. One pod runs three sequential four-rank processes so each arm releases its model state before the next while retaining the same node/GPU allocation. Each process loads only the checkpoint-100 student from DCP; rank-16, rank-128, and full-weight critics all start from the identical pretrained teacher function with fresh optimizer state.
- Every rank deterministically regenerates 16 training and 16 disjoint held-out contexts; rank `r` maps to interval `r`. Cross-arm full-bank tensor fingerprints must match exactly before the summary is accepted. Critic training uses unit weights to remove SNR allocation as a capacity confound. Student parameters are used for the gradient diagnostic but never stepped.
- Arms are current attention-only rank 16/alpha 32, the same targets at rank 128/alpha 256, and full weight. Each receives 50 critic-only updates with evaluations at 0/1/10/25/50. Rank 16 uses `8e-6`; higher-capacity LRs are calibrated from training context 0 to match the rank-16 first functional critic-output RMS on held-out context 0, with a 25% acceptance bound. AdamW betas, weight decay, and clipping remain `[0.0,0.999]`, `0.01`, and `1.0`.
- Each evaluation records 16-context train/held-out target RMS, held-out update-to-target alignment, effective parameter delta, and actual-versus-sample-oracle local student-gradient cosine on four held-out contexts per interval. The predeclared capacity gate is the plan's >=10% held-out advantage in >=3/4 intervals including interval 3, interval-3 cosine gain >=0.2 and positive, plus real held-out improvement.
- Python AST, shell syntax, Kubernetes server dry run, and `git diff --check` passed. The manifest requests/limits exactly four GB200 GPUs on one node. SHA-256: harness `8824e92950de2e7fb758ddc34794def96a9eedf97eaffe05b8663e744084b2b7`; runner `48d14b1068def0d91dc9e798a64b47d31084e842129c47ab71c65952c7846ca9`; manifest `e8b1c7c5c4cfea679902b3e54a2d46068985ec4b3f2c412ac985359d29220756`. Create an immutable ConfigMap, submit, and preserve/checksum all arm JSONs, summary, and logs before releasing the pod.

### 2026-09-16 critic-capacity gate submitted

- Created immutable ConfigMap `tdm-critic-capacity-6676ef6-r1-assets` and submitted pod `vllm/tdm-critic-capacity-6676ef6-r1`.
- It scheduled immediately on node `10.0.129.200`, exposing exactly four NVIDIA GB200 GPUs (189471 MiB each), Torch `2.12.0+cu130`, and CUDA 13.0. It detached exact commit `6676ef6b1`, passed the in-container AST gate, and began the rank-16 arm at `2026-09-16T19:41:47Z`.
- Monitor arm logs and terminal markers. Do not start a coupled training continuation until all three arms pass fingerprint/calibration validation and the predeclared capacity gate is evaluated.

### 2026-09-16 critic-capacity rank-16 arm complete

- The rank-16 reference arm completed cleanly at `2026-09-16T19:50:01Z` after about 8 minutes 14 seconds. Selective checkpoint-100 student loading, deterministic bank generation, 50 critic-only updates, and all scheduled evaluations succeeded.
- Its held-out-context-0 first functional update RMS was `0.004869403887009539`, essentially matching the independently recorded rank-16 calibration target `0.004852754529565573`. Trainable parameter count is `11,796,480`, selected learning rate is the planned `8e-6`, and `rank16.json` is present on the retained volume.
- The orchestrator immediately started the rank-128 arm at `2026-09-16T19:50:01Z`. Continue monitoring its calibrated learning-rate selection and completion before interpreting capacity.

### 2026-09-16 critic-capacity rank-128 arm complete

- The rank-128 same-target-module arm completed cleanly at `2026-09-16T19:58:09Z`, about 8 minutes 8 seconds after launch. It has `94,371,840` trainable parameters.
- Functional calibration selected learning rate `7.443495515002623e-6`; achieved step-1 held-out-context-0 output RMS was `0.005189572015630405` versus target `0.004852754529565573` (`1.0694x`), inside the predeclared `0.75-1.25x` bound. `rank128.json` is present on the retained volume.
- The orchestrator started the full-weight critic arm at `2026-09-16T19:58:09Z`. Do not interpret or continue until it and the cross-arm summary validations finish.

### 2026-09-16 critic-capacity r1 calibration rejection

- The full-weight arm itself completed all 50 updates/evaluations at `2026-09-16T20:06:14Z`, with `1,418,996,800` trainable parameters and selected learning rate `8.98790629532629e-8`.
- Its actual step-1 held-out output RMS was `0.006372954330486093` versus target `0.004852754529565573`, a `1.3133x` ratio. This is outside the predeclared `0.75-1.25x` functional-calibration acceptance band. The cross-arm postprocessor correctly rejected r1 and the pod exited `Error`; do not interpret its capacity gate.
- Root cause is confined to the temporary calibration harness: it performs a base trial and one verification, applies a second proportional learning-rate correction when needed, but returns that corrected rate without verifying it. The final full-arm correction landed nonlinearly outside tolerance.
- Preserve r1 JSON/logs as invalid calibration evidence. Prepare an r2 full-only retry that reuses r1's accepted rank-16/rank-128 reports and exact target, iterates/validates the full-arm calibration before the 50-update run, then runs the unchanged cross-arm fingerprint/calibration/gate postprocessor. Do not widen the tolerance or rerun the already valid LoRA arms.

### 2026-09-16 critic-capacity full-arm r2 submitted

- Prepared the narrow full-only retry. Its calibration now proportionally iterates for up to eight verified trials and proceeds only after a trial lands within `0.98-1.02x` of the unchanged rank-16 target; failure to converge aborts before the 50-update arm. The cross-arm postprocessor still enforces the original `0.75-1.25x` bound and the original scientific gate.
- r2 reuses byte-copied `rank16.json` and `rank128.json` from retained r1 and records their source hashes. It deterministically regenerates only the full arm, after which fingerprint equality against both reused arms is required.
- Python AST, `bash -n`, and Kubernetes server dry run passed. SHA-256: harness `1bbc89f2b1bea27bb7dd7d80ebe0d0b75ae1bc334dddf20c327e0fd915886b32`; runner `51957cc936a102b3500900c7a7c009941196af65f3776531bfda6f2ac07f71f9`; manifest `c1e7a045db3791edd7fd02fe7c939b247abadee6a09a84a5d34c8dad5622706e`.
- Deleted only the failed r1 pod after confirming its volume evidence is retained. Created immutable ConfigMap `tdm-critic-capacity-6676ef6-r2-assets` and submitted pod `vllm/tdm-critic-capacity-6676ef6-r2`, still requesting exactly four GB200 GPUs on one node.

### 2026-09-16 critic-capacity r2 calibration floor

- r2 aborted before the 50-update arm because the deliberately stricter `0.98-1.02x` calibration criterion did not converge. This is a harness calibration result, not a capacity result; no new full-arm report or cross-arm gate was produced.
- Eight verified full-weight trials showed a low-learning-rate output-delta floor/non-monotonicity consistent with BF16 parameter quantization. Output RMS moved from `0.0087571` at `1.6516e-7` to a best observed `0.00542745` at `3.25055e-8`, then rose to `0.00549718` at `2.90636e-8`. The best trial is `1.1184x` the `0.00485275` target and therefore inside the original predeclared `0.75-1.25x` acceptance band, even though a newly imposed 2% target is unattainable in this representation.
- Prepare r3 to retain all eight verified trials, select the closest verified learning rate only if it falls inside the unchanged original `0.75-1.25x` band, and otherwise abort. This does not weaken the planned scientific calibration rule; it removes the extra 2% condition added only for the retry. Reuse r1's accepted rank-16/rank-128 reports again and rerun only the full arm.

### 2026-09-16 critic-capacity full-arm r3 submitted

- r3 retains the eight verified calibration trials, selects the closest trial by absolute log-ratio only from candidates within the original `0.75-1.25x` band, and aborts if none qualify. It still regenerates only the full arm and applies the unchanged cross-arm gate.
- Python AST, `bash -n`, and Kubernetes server dry run passed. SHA-256: harness `1f97a010762805fa621fd5c1f8933d80c8d0603790ad439b890fe676a9acfbd0`; runner `65b3ce63c9670f2e771e77c471bbf09a032181cd936d347be51764d447b88700`; manifest `a0fcd885b3d8684125bf7ab3e9ac3712150b5492d654f898c396feed99aefda4`.
- Deleted only failed r2 pod after preserving its volume evidence, created immutable ConfigMap `tdm-critic-capacity-6676ef6-r3-assets`, and submitted `vllm/tdm-critic-capacity-6676ef6-r3` with the unchanged exact four-GB200/one-node request.

### 2026-09-16 critic-capacity gate completed negative

- r3 completed successfully at `2026-09-16T20:26:28Z` on exact signed commit `6676ef6b1`, one node, and exactly four GB200 GPUs. Cross-arm deterministic bank fingerprints matched, all three arms began at the teacher function, the checkpoint-100 student was never stepped, and the rank-16/rank-128/full calibration ratios were `1.0034x`, `1.0694x`, and `1.1079x`, all inside the predeclared `0.75-1.25x` band.
- The predeclared capacity gate is negative for both higher-capacity arms. Relative to rank 16 at step 50, rank 128 held-out RMS improvements by interval were `11.264%`, `1.899%`, `1.470%`, and `0.693%`; only interval 0 met 10%, and interval-3 generator-gradient cosine worsened from `+0.17943` to `-0.11917` (gain `-0.29860`). Full-weight improvements were only `0.657%`, `0.664%`, `0.410%`, and `0.148%`; interval-3 cosine worsened to `+0.14655` (gain `-0.03288`). Neither arm met any multi-interval capacity criterion.
- Final step-50 held-out RMS for rank16/rank128/full was respectively: interval 0 `0.227629/0.201988/0.226134`; interval 1 `0.088549/0.086867/0.087961`; interval 2 `0.064455/0.063508/0.064191`; interval 3 `0.049307/0.048966/0.049234`. Rank 128 has a genuine early-interval benefit but does not repair the hard final interval and actively degrades its generator direction.
- Do not launch a higher-capacity checkpoint-100 continuation or coupled TDM run. The evidence does not support current attention-rank-16 critic capacity as the dominant blocker. The next discriminator is the already identified norm-matched weighting/update-direction control.
- Preserved all r1/r2/r3 evidence locally under `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-critic-capacity-6676ef6-r{1,2,3}/`; all 31 local files match the retained PVC hashes. Key r3 SHA-256: `summary.json` `d02122c50ae79ef67fb32dd10f4d3219f88d2efda462caa86a9d4ffa52739b54`; `rank16.json` `ae1e71a6c994f91b9106c1f5ac350d507d2729803fd3a4c528300391d413c57b`; `rank128.json` `2bfa91be25b5522428b0aca9431647fc667cd92438dc0f3330ea27a7119a1efb`; `full.json` `effd1dc50d24f1b2929c544e9344e74d2bd06628f4298de94a5f354151c5c704`.
- Deleted only the completed r3 pod after local/PVC checksum verification, releasing all four GPUs; `kubectl get pods -l issue=775` now reports no resources. Retained the PVC evidence and immutable r1/r2/r3 ConfigMaps.
- Final GitHub check as `macthecadillac`: issue 775 remains open and assigned with its same two comments and no proposed fix; no open TDM/775 PR exists. Fork branch and clean code tip remain exact signed commit `6676ef6b10ae240f0cb62dc8bf3d959faf777819`; upstream main remains `9b0e57fe4b3a8c112ff24cc22f752eb0608ccfab`. The only worktree modification is this mandatory local handoff.

### 2026-09-16 real-score-conditioning diagnostic prepared

- User authorized the cheaper update-free real-score-conditioning diagnostic before the norm-matched weighting/update-direction control. Re-read all applicable `AGENTS.md` files, the modular trainer guide, relevant TDM/CFG documentation, and the `launch-experiment` skill. The skill's generic experiment-journal instruction remains superseded by current repository policy, so active state remains in this mandatory handoff.
- Preflight passed: `gh` is authenticated as `macthecadillac`; issue 775 is still open/assigned with the same two comments and no proposed fix; no open TDM/775 PR exists; local and fork branch tips both equal signed commit `6676ef6b10ae240f0cb62dc8bf3d959faf777819`; the only worktree modification is this handoff; and no issue-775 pod is active in namespace `vllm`.
- Prepared temporary job `tdm-real-score-conditioning-6676ef6-r1`. It fully loads and asserts nonzero student and critic LoRA parameter deltas from the retained coupled/paper-pseudo-Huber checkpoint 100, then disables optimizer steps. Rank `r` is fixed to interval `r` for schedule `[1000,750,500,250]`. It deterministically regenerates the capacity diagnostic's held-out bank (16 contexts/interval, trajectory seed `510000`, context seed `610000`) and computes generator gradients on contexts 0-3.
- Each context computes the checkpoint critic's positive-conditioned fake prediction exactly once. The same trajectory, source prediction, production flow transition, target-noisy latent, timesteps, proposal noise, positive conditioning, and cached critic prediction are reused for `negative_g6`, real-encoded `empty_g6`, and conditional-only `conditional_g1`. The harness calls production `predict_x0`, flow transition helpers, and the production paper pseudo-Huber surrogate; it does not implement an approximate score conversion.
- The report includes raw conditioning tensor fingerprints/hashes and pairwise embedding/mask RMS, aborts on zero placeholders or identical negative/empty conditioning, records all requested per-context tensor/gradient/oracle metrics, summarizes medians, and evaluates the exact predeclared interval-3 qualification gate. `conditional_g1` has no unconditional dependence in its real target; its applied CFG correction is recorded as zero.
- Python AST, `bash -n`, Kubernetes server dry-run, `git diff --check`, and the final no-conflicting-pod check passed. The manifest requests and limits exactly four `NVIDIA-GB200` GPUs on one node and retains PVC evidence plus immutable ConfigMap provenance. SHA-256: harness `792fde8e0beff4b34af51b7687a253f1a8536246a28d50bb5ea5ca4a034b2c5f`; runner `a9ab12486104b786b3c78b2d7d5a05c87c5cdc803d702d17c0cb22492baada50`; manifest `52e043a4fb87da3348d61fdaf66ea2fa0fb0ec31b4b6775e266a6fdc0e26ac1a`.
- Next: create immutable ConfigMap `tdm-real-score-conditioning-6676ef6-r1-assets`, submit the one-node/four-GB200 pod, monitor to `.run_done`, analyze the predeclared gate, copy every JSON/input/log/status artifact locally with PVC SHA-256 verification, update this handoff, delete only the completed pod, and recheck no pod/branch/GitHub changes. Do not launch training automatically.

### 2026-09-16 real-score-conditioning diagnostic submitted

- Created ConfigMap `tdm-real-score-conditioning-6676ef6-r1-assets` and sealed it immutable (`resourceVersion 396638077`) before pod submission. Submitted pod `vllm/tdm-real-score-conditioning-6676ef6-r1` with the exact four-GB200 request/limit.
- The pod scheduled immediately on node `10.0.129.200`, exposes exactly four NVIDIA GB200 GPUs with 189471 MiB each, reports Torch `2.12.0+cu130`/CUDA 13.0, detached exact commit `6676ef6b1`, passed the in-container AST gate, and began the four-rank diagnostic at `2026-09-16T21:03:50Z`.
- Monitor context-completion markers and `.run_done`. Do not launch any training control after the gate.

### 2026-09-16 real-score-conditioning r1 harness-order failure

- r1 stopped before checkpoint loading, context generation, or metric computation because its schedule assertion read `_denoising_step_list` before the first student trajectory lazily initialized that production cache. All four ranks raised the same `AttributeError`; no scientific result was produced and no model or optimizer update occurred.
- Preserved the complete seven-file failed-run evidence under `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-real-score-conditioning-6676ef6-r1/`. Every local artifact matches its PVC SHA-256, including diagnostic log `579341666c77fbb1a2987b09e3b06a4094a5b90f514ae68955cf576d9de38a97` and orchestrator log `9161730761038e9a868de9a12c01aa93a99c912ba2d8540ec9543498b07d5bdc`.
- Prepared narrow r2: move only the schedule-value assertion until after `build_context_bank` initializes the schedule, retaining the expected `[1000,750,500,250]` assertion. No data, model, checkpoint, seed, conditioning, metric, or gate behavior changed. AST, shell syntax, and Kubernetes server dry-run pass. SHA-256: harness `6ab7551b9675f5e5a7c93bd16101dce43ca92423f73f89892fba5561d7f6462a`; runner `f2f1592bebc448d35297c663d98d9bc3b2cdc953bd3293f2e17487087a7f692d`; manifest `f827d235b857ae552a28cdd821d4ac49f6b8a7f3e03ba9cec7ce2c447d88e6b4`.
- Delete only failed r1 after its evidence verification, then create immutable r2 provenance and submit the unchanged four-GB200 diagnostic.

### 2026-09-16 real-score-conditioning r2 mask-comparison failure

- r2 loaded the full checkpoint and generated the deterministic context bank, then stopped before teacher/critic score or gradient evaluation because the fingerprint-only comparator directly subtracted real attention masks with different genuine lengths (positive 512 versus production-negative 98). This is a temporary reporting-harness bug, not a model-input failure; no optimizer step or scientific metric was produced.
- Preserved all seven r2 files under `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-real-score-conditioning-6676ef6-r2/`; every local file matches its PVC SHA-256. Diagnostic log SHA-256 is `dc65655f17a7b8dce8ccbceed14a88eb10b493ede21433bd4f5e55bf63650871`; orchestrator log is `93997e0c05651f21c22f9a674fce34f5549dbdc4d025d01dca2ae598a831f400`.
- Prepared isolated r3 fix: fingerprint pairwise RMS now flattens each original tensor and zero-pads only temporary comparison copies to the larger element count. Original embeddings/masks, their shapes, fingerprints, and the genuine masks supplied to teacher forwards are untouched. The report states this definition explicitly. No experimental behavior changed.
- r3 AST, shell syntax, and Kubernetes server dry-run pass. SHA-256: harness `1eed7219f7f59eefdcc8c2f47b4de268f45aeab6cd9286844d5f595fc1e94f81`; runner `f7e46db54fe0bb0de15be6737c71c81493d88047ed95e57a169439ece8eb3c91`; manifest `35abb27b7655a406499798c3ec517d59abf0f2c556c8a39109b3c1c457354597`.
- Delete only failed r2 after evidence verification, then submit r3 with immutable ConfigMap provenance and the unchanged exact-four-GB200 constraint.

### 2026-09-16 real-score-conditioning r3 compiled-backward failure

- r3 successfully loaded both checkpoint roles, regenerated the bank, accepted the native real empty-prompt mask, and reached gradient context 0. It stopped on the first retained-graph `torch.autograd.grad`: Wan's compiled backward uses donated buffers and requires `retain_graph=False`. No optimizer step occurred and no complete context/report was emitted.
- Preserved all seven r3 files under `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-real-score-conditioning-6676ef6-r3/`; every local file matches its PVC SHA-256. Diagnostic log SHA-256 is `257d2b0b694d7bcd385d27be541482157f21fc7b6eb085a59ea021955c7bbb4e`; orchestrator log is `d713fb58e58f64f8b82e99252618d354efb4d72f7c6fe3046127b7594378a72e`.
- Prepared r4 to use the production `WanModel.backward` exactly once per fresh graph. For each of the six actual/oracle gradients on contexts 0-3 it reruns only the student source forward over the identical source latent, timestep, positive conditioning, and attention metadata; it asserts every regenerated prediction is bit-exact to the shared baseline before backward. The target-noisy latent, proposal noise, cached one-call critic prediction, all teacher predictions, and variant targets remain shared. Optimizer steps remain trapped/forbidden.
- r4 postprocessing requires six source recomputations with maximum absolute difference exactly zero on each gradient context and none on contexts 4-15. AST, shell syntax, and server dry-run pass. SHA-256: harness `5cc13396f06e62dbe0207fd75f23335a3939c20e01baaa161ff60e577971da30`; runner `7bf50d60606834182c98600ab1778be39a01e89218bd09a794a0c99e60caaed9`; manifest `7911abe5da4f2b0e8e5993ed8231af46c15564eb14c5828e37b9f197239efd7a`.
- Delete only failed r3 after evidence verification, then submit r4 with immutable provenance. The scientific design and gate remain unchanged.

### 2026-09-16 real-score-conditioning gate completed negative

- r4 completed successfully (`rc=0`) at `2026-09-16T21:27:31Z` on node `10.0.129.200`, exact commit `6676ef6b1`, and exactly four GB200 GPUs. All 64 contexts completed: 16 per interval, with gradients on predeclared contexts 0-3. No optimizer step was possible because both optimizer `step` methods were trapped. Every one of the six source-forward recomputations on each gradient context was bit-exact to its shared baseline (`max_abs_difference=0`). The checkpoint load changed both roles on every rank by student LoRA L2 `50.6365681` and critic LoRA L2 `50.6291834`; post-load role hashes were replica-identical.
- The generated bank exactly matches all common fingerprint values from the capacity diagnostic's retained held-out bank across all 64 contexts after normalizing only renamed/new fingerprint fields. Design remained schedule `[1000,750,500,250]`, rank `r` to interval `r`, trajectory seed `510000` with stride `10003`, and context seed `610000` with stride `20011`.
- Conditioning fingerprints were identical across ranks and nonzero. Positive embedding/mask SHA-256: `16f3c2eb...` / `ad755ff2...` (mask shape `[1,512]`, 512 nonzeros). Production-negative: `49689a88...` / `b854b999...` (mask `[1,98]`, 98 nonzeros). Encoded empty: `3fb0f077...` / `b9c205bd...` (mask `[1,1]`, one nonzero). Production-negative versus empty padded-comparison RMS was `0.0348081` for embeddings and `0.994885` for masks, so they are neither identical nor zero placeholders. Comparison padding affected fingerprints only; genuine native masks were supplied unchanged to the teacher.
- Interval median actual-to-corresponding-sample-target-oracle gradient cosine for `negative_g6 / empty_g6 / conditional_g1` was: interval 0 `+0.96237 / +0.97866 / +0.03725`; interval 1 `+0.90742 / +0.88858 / -0.46295`; interval 2 `+0.43519 / +0.42262 / -0.35616`; interval 3 `-0.29569 / -0.31886 / -0.59496`.
- Corresponding median actual gradient norms were: interval 0 `2039.89 / 2886.27 / 3336.04`; interval 1 `219.28 / 193.16 / 592.74`; interval 2 `63.84 / 66.49 / 121.45`; interval 3 `13.20 / 12.80 / 19.34`. Thus both candidates passed only the broad interval-3 norm-ratio guard (`0.969x` empty, `1.465x` conditional).
- All-16-context median real-minus-fake RMS for `negative_g6 / empty_g6 / conditional_g1` was: interval 0 `0.41853 / 0.34099 / 0.02187`; interval 1 `0.22033 / 0.22944 / 0.01702`; interval 2 `0.11407 / 0.10370 / 0.01422`; interval 3 `0.04452 / 0.04358 / 0.01429`. The applied CFG correction accounts for about `97.65%/98.39%`, `97.72%/98.38%`, `96.36%/96.72%`, and `91.17%/91.04%` of negative/empty total delta RMS by interval, so CFG dominates the latent delta magnitude but removing it does not repair gradient direction.
- At interval 3, production context cosines were `[-0.05240,-0.53897,-0.53956,+0.24466]`. Empty-prompt values were `[-0.15582,-0.53268,-0.48190,-0.06065]`: median worsened by `-0.02318`, only 2/4 contexts improved, and 0/4 were positive. Conditional-only values were `[-0.57971,-0.85757,-0.61021,-0.37597]`: median worsened by `-0.29928`, 0/4 improved, and 0/4 were positive. Empty had no >0.10 early-interval median regression (changes `+0.0163/-0.0188/-0.0126`), while conditional regressed intervals 0-2 by `-0.925/-1.370/-0.791`.
- Neither non-production variant qualifies. Both target oracles remain strongly aligned with the production oracle at interval 3 (median oracle-gradient cosine `0.98134` empty and `0.97370` conditional), so their failure is not explained by selecting a radically different sample target. Empty versus production actual gradients are less aligned (`0.78451` median at interval 3), while conditional versus production actual gradients are `0.68676`.
- Scientific decision: production negative-prompt semantics and CFG strength are unlikely to be the primary defect. Do not launch an empty-prompt training control, a guidance sweep, or any 100-update run from this result. Proceed next to the norm-matched critic-weighting/update-direction control; do not return to the completed capacity ceiling.
- Preserved all eleven r4 artifacts under `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-real-score-conditioning-6676ef6-r4/`; every local file matches its PVC SHA-256. Key hashes: `diagnostic.json` `33bd1df85b5c31a871d440be038ad0d7fa4aa93047310115a34a058b82af8024`; `summary.json` `fc992a470b1f095aa5c440f873aa5d4bbc089f785dff6362b2a7ea5ccd3c158c`; diagnostic log `ae189a9ffd79ca9cde5f11d089d80b8cc1525e52a87327e52a565a42710cc534`; orchestrator log `217e31c9003a69521e11f7456dbf969e010a06c435158d867bf2c0b2e5a03be9`; harness `5cc13396f06e62dbe0207fd75f23335a3939c20e01baaa161ff60e577971da30`; runner `7bf50d60606834182c98600ab1778be39a01e89218bd09a794a0c99e60caaed9`. Retain the PVC run and immutable ConfigMaps r1-r4. Delete only the completed r4 pod after final state checks and release all four GPUs.
- Final cleanup/state: deleted only completed pod `vllm/tdm-real-score-conditioning-6676ef6-r4` after the eleven-file checksum comparison; `kubectl get pods -l issue=775` reports no resources, so all four GPUs are released. PVC runs and all immutable r1-r4 ConfigMaps remain.
- Final GitHub/worktree verification as authenticated `macthecadillac`: issue 775 remains open, assigned to `macthecadillac`, with two unchanged comments; no open PR exists for `macthecadillac:issue/775-tdm`; local and fork tips both remain exact signed commit `6676ef6b10ae240f0cb62dc8bf3d959faf777819`; `git diff --check` passes; the only worktree change is this mandatory untracked-state handoff modification. No production code, branch, commit, or GitHub state changed during the diagnostic.
- Sent the required completion notification successfully through `/home/sandbox/notify` with title `TDM diagnostic complete`.

### 2026-09-16 handoff durability policy changed

- At the user's direction, the machine-wide FastVideo handoff rules now require active pre-PR handoffs to be tracked, committed, and pushed on the dedicated workload branch so they survive reboots and loss of temporary worktrees.
- This handoff is being committed and pushed as branch-local active-work state. It remains temporary: before opening a draft PR or pushing any update to an existing PR, transfer the relevant context to the PR body, remove every tracked `.agents/handoffs/*` file with `git rm`, commit and push that deletion, and verify that no tracked handoff remains in the branch.
- The scientific state is unchanged: the real-score-conditioning gate was negative, no issue-775 Kubernetes pod is active, and the next experiment is the norm-matched critic-weighting/update-direction control.
- Durability-commit preflight: `git diff --check` passes; `gh` is authenticated as `macthecadillac`; issue 775 remains open and assigned with its same two comments; no open TDM/775 PR exists; and the fork tip remains `6676ef6b1`. The configured local pre-commit command could not run because `pre-commit` is not installed on this host; this commit changes only the Markdown handoff.

### 2026-09-16 norm-matched critic-weighting diagnostic prepared

- User authorized the next discriminator. Re-read the complete handoff, current repository/modular-training instructions, and the `launch-experiment` skill. The skill's legacy experiment-journal instruction remains superseded by repository policy, so all state stays in this tracked handoff. Preflight passed as authenticated `macthecadillac`: issue 775 remains open/assigned with the same two comments, no open TDM/775 PR exists, local/fork tip is handoff-only commit `fefa13abd` over production-code tip `6676ef6b1`, upstream main remains `9b0e57fe`, the worktree was clean, and no issue-775 Kubernetes pod was active.
- Prepared temporary job `tdm-norm-matched-weight-6676ef6-r1`; production code is unchanged and the job checks out exact signed code commit `6676ef6b10ae240f0cb62dc8bf3d959faf777819`. It loads both frozen checkpoint-100 roles from `/workspace/run/issue-775/tdm-coupled-pseudohuber-100step-r1/output/checkpoint-100`, asserts that loading changes both adapters, and asserts at exit that the student never changed and the critic was exactly restored.
- The diagnostic reuses the prior mixed-weighting training seeds `71000 + 10000*r` for four deterministic replicas and validates their rank/interval weight metadata exactly against retained report SHA-256 `eb3d231b8bb9afda1d6fa14184cfb1d2c588d35d0ee7be1497aab836d5ffcf55`. Rank `r` remains fixed to interval `r`. Each replica evaluates four disjoint held-out trajectories with seeds `131000 + 10000*r + 1000*h`, giving 16 held-out contexts per interval; context tensors receive deterministic SHA-256 fingerprints and all three arms share each in-memory held-out context.
- From an identical checkpoint critic/optimizer/scheduler snapshot, production, mean-matched importance-only, and mean-matched unit weighting each take one exact production `_finish_role_update` AdamW step. The harness captures the represented parameter delta for every arm, then rescales and reapplies each alternative direction to the checkpoint snapshot until its represented L2 norm matches the production delta within `0.5%`, accounting for dtype rounding. It records raw/matched norms, scale trials, raw-delta cosines, training error, held-out critic target RMS/update cosine, and actual generator-gradient cosine to the same sample-target oracle. No student optimizer step occurs.
- Predeclared candidate gate: norm error <=`0.5%` in every replica; held-out critic update-cosine median gain >=`0.02` in at least 3/4 intervals; better median held-out critic RMS than production in at least 3/4 intervals with no interval worse than `1.001x`; interval-3 generator-to-oracle median gain >=`0.01` with at least 10/16 contexts improved; and no interval 0-2 median generator-cosine regression >`0.01`. Only a qualifying candidate warrants a later 100-update training control; this job must stop after the diagnostic.
- Kubernetes server dry run passed. The manifest requests and limits exactly four `NVIDIA-GB200` GPUs in one pod/node, retains PVC evidence, and will use an immutable ConfigMap. No local tests or harness execution were performed. SHA-256: harness `bec5204534485584131c3cea017db50d58df42cf751cf02dcadd4aa08e74aff0`; runner `c23cd1c9976bff65bbeee9207bec82b10f946ae129db3ecafed7a3887c627711`; manifest `f17f0bd596080d76290f975cf06121b9da7d6583950eeab78f882125ebbfdce0`.
- Next: GPG-sign and push this handoff update, recheck no conflicting pod, create and seal immutable ConfigMap `tdm-norm-matched-weight-6676ef6-r1-assets`, submit the pod, monitor to `.run_done`, evaluate the predeclared gate, checksum-copy all report/input/log/status artifacts locally, update and push this handoff, then delete only the completed pod. Do not launch a training control automatically.

### 2026-09-16 norm-matched diagnostic submitted

- GPG-signed and pushed the preparation handoff as `88a8da973`. Created ConfigMap `tdm-norm-matched-weight-6676ef6-r1-assets`, sealed it immutable at resource version `396744484`, and submitted pod `vllm/tdm-norm-matched-weight-6676ef6-r1`.
- The scheduler placed the unchanged request=limit four-GB200 pod intact on node `10.0.140.245`. The container is still pulling the development image; no harness result exists yet. Continue monitoring startup, on-cluster AST/GPU/checkpoint gates, replica completion, and `.run_done`.

### 2026-09-16 norm-matched r1 shallow-clone failure

- r1 exposed exactly four GB200 GPUs and the expected Torch/CUDA stack, then stopped before checkout, on-cluster AST parsing, model/checkpoint loading, context generation, or optimizer work. The branch had advanced by two handoff-only commits after the runner was rendered, so `git clone --depth 1` did not contain exact production-code commit `6676ef6b1`; detached checkout failed with `fatal: reference is not a tree`. No scientific result exists.
- Retain r1's PVC `status` and orchestrator log. Prepared isolated r2 changing only clone depth from 1 to 8 so exact `6676ef6b1` is available; the harness, reference report, experimental seeds, metrics, and predeclared gate are unchanged. r2 server dry run passed with the same exact-four-GB200 request. SHA-256: harness `bec5204534485584131c3cea017db50d58df42cf751cf02dcadd4aa08e74aff0`; runner `a844c4b5f9bb3d68e19056eac4f94074ce6d7327c50827fbc12a64673f491f82`; manifest `3a5b583289f4e68b219aafacc0d3dd00ff9c044f6df0bfc414f59db4d8cd5ea0`; reference `eb3d231b8bb9afda1d6fa14184cfb1d2c588d35d0ee7be1497aab836d5ffcf55`.
- Commit and push this state, create/seal `tdm-norm-matched-weight-6676ef6-r2-assets`, submit r2, preserve both r1 and r2 evidence, and delete failed r1 only after local/PVC verification.

### 2026-09-16 norm-matched diagnostic r2 submitted

- GPG-signed and pushed the r1 failure/retry handoff as `310c98055`. Created and sealed immutable ConfigMap `tdm-norm-matched-weight-6676ef6-r2-assets` (`resourceVersion 396758145`) and submitted `vllm/tdm-norm-matched-weight-6676ef6-r2` with the unchanged exact-four-GB200/one-node request.
- r2 scheduled on node `10.0.140.245` with four NVIDIA GB200 GPUs and the same development image digest `sha256:0a3c9840054ea4bc0e819cba216d41cf5ee90631f7447e44dc4a42946aa09fef`. It reported Torch `2.12.0+cu130`/CUDA 13.0, checked out exact production-code commit `6676ef6b1`, passed the in-container AST gate, and began the four-rank diagnostic at `2026-09-16T22:50:20Z`.

### 2026-09-16 norm-matched critic-weighting gate completed negative

- r2 completed successfully at `2026-09-16T23:00:17Z` with `rc=0`. Both checkpoint-100 roles loaded on all four ranks (student LoRA load delta L2 `50.6165011`, critic `50.6115380`), student optimizer steps remained zero, and exact final student and restored critic deltas were both zero. Training metadata matched the retained mixed-weighting report exactly; all 16 training and 64 held-out context fingerprints were unique, with no overlap.
- The represented AdamW parameter-delta match was far tighter than the predeclared `0.5%` bound: maximum relative errors were `7.45e-8` for importance-only and `7.79e-8` for unit. Before matching, both alternative deltas were much larger than production: importance-only ratios by replica were `14.3740x`, `14.1948x`, `13.2588x`, and `36.1007x`; unit was `14.3762x`, `14.1988x`, `13.2581x`, and `36.1187x`. Importance-only and unit raw directions were almost identical (cosine `0.999972-0.999994`), while their cosine to production ranged from `-0.00869` to `+0.12438`.
- Neither alternative passed the predeclared gate. Paired median held-out critic update-cosine gains for importance-only by interval 0-3 were `+0.01531`, `+0.00674`, `+0.00082`, `-0.00155`; unit was `+0.02333`, `+0.00015`, `+0.00067`, `-0.00120`. Thus importance-only cleared the `+0.02` directional threshold in 0/4 intervals and unit in only 1/4, versus the required 3/4.
- Paired median critic RMS ratios to production for importance-only were `0.999584`, `0.999561`, `0.999939`, `1.000100`; unit was `0.999265`, `0.9999997`, `0.999955`, `1.000086`. Both technically improved 3/4 intervals and stayed inside the `1.001x` maximum, but the changes were only about `0.0000-0.0735%` and both worsened interval 3.
- Absolute median held-out update-to-target cosines for production / importance-only / unit were: interval 0 `0.01989 / 0.03281 / 0.03690`; interval 1 `0.03298 / 0.03472 / 0.02855`; interval 2 `0.03912 / 0.04840 / 0.04517`; interval 3 `0.03790 / 0.03531 / 0.03554`. Absolute median post-update generator-gradient cosine to the sample-target oracle was: interval 0 `0.96549 / 0.96503 / 0.96489`; interval 1 `0.91037 / 0.91117 / 0.91128`; interval 2 `0.80802 / 0.80983 / 0.80987`; interval 3 `0.08962 / 0.08746 / 0.08753`.
- On the paired generator-gradient gate, interval-3 median gains were `-0.000296` for importance-only and `-0.000870` for unit; only 7/16 contexts improved for each, versus the required gain `>=0.01` and at least 10/16. Early-interval regression remained safely below `0.01` (`0.000280` maximum importance-only, `0.000517` unit), but that does not rescue either candidate.
- Scientific decision: norm-matching removes the prior step-size confound and confirms that neither removing only clipped-SNR weighting nor using unit weighting provides a useful one-step direction at the production update norm. The tiny early-interval critic changes do not repair interval 3 and do not justify a 100-update weighting control. Combined with the negative capacity and real-score-conditioning gates, the next experiment should target a different mechanism; do not return to the capacity ceiling or train either weighting candidate from this result.
- Preserved r1's two failed-launch files under `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-norm-matched-weight-6676ef6-r1/` and all ten r2 result/provenance files under `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-norm-matched-weight-6676ef6-r2/`. Every copied file matches its PVC SHA-256 exactly. r1 hashes: orchestrator log `89ceb14e80ef047b84719ea1f091df0a16bbe72cd6baf36d36db174c104d8ac9`; status `3fdf6b39f9186a2f7149d9bb28ad5abeaf901bff205bf8678db2667ad4212393`. r2 hashes: `diagnostic.json` `a66e1128b72da9947bd18cd78caaf3db99bc5f4068c603e9fa5e7fe5cebbe902`; `summary.json` `485db9231b0f982d7de2e08313311591db8e59cc1de7aa5f1d4ad74abc842ce3`; diagnostic log `ce9e56af82e7655713df5eba2ffd83666db0219c119b86b3a6fe2fbb3d22a8fa`; orchestrator log `fbfff24a8e94e1600c39b853af6e3d900af2fd0acb540efa9f761b808e910457`; `commit.txt` `31ae1a99f65fb2d45b271bfd343a0faaa7087e099afe4625785e08fb2e65c542`; `input-sha256.txt` `e19c82bbcbbfa66ab5150fa23071fe3199a2a95231d2ba7e7bb76c398b03f7bb`; `result.txt` `772138cd2e0ec32dbd14b113aebb326226c4d609b7d4720b457a264183081526`; status `bc67ce5f86f21f579280ee3d91c54bb8243a3776c536ca8e219b78f94c01dc4a`; empty `.complete` and `.run_done` each `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Immutable ConfigMaps r1 (`resourceVersion 396744484`) and r2 (`396758145`) plus both PVC run directories are retained. After this handoff update, delete only the failed r1 and completed r2 pods, confirm no issue-775 workload remains, verify branch/GitHub state is unchanged, and send the mandatory completion notification. No follow-on training job was launched.


### 2026-09-16 norm-matched diagnostic final cleanup

- Deleted only failed pod `vllm/tdm-norm-matched-weight-6676ef6-r1` and successfully completed pod `vllm/tdm-norm-matched-weight-6676ef6-r2` after all 12 retained files matched their PVC hashes. `kubectl get pods -n vllm -l issue=775 -o name` is empty, so the four GB200 GPUs are released. Both immutable ConfigMaps and PVC evidence directories remain.
- Final read-only GitHub check as authenticated `macthecadillac`: issue 775 remains open and assigned to `macthecadillac` with its same two comments; the open-PR search for `775 OR TDM` is empty. No GitHub state was changed.
- Local and fork branch tips both remain `310c98055fb48906f93fb053148d0c4214d2e67c` before this final handoff durability commit, and `git diff --check` passes. The only worktree change is this mandatory tracked handoff update. No production code changed, no PR/review loop began, and no follow-on training control was submitted.


### 2026-09-17 Wan student-reachability ceiling authorized

- User authorized the conditional sequence: run the Wan reachability ceiling first; only if Wan fails, move to MiniMax H3 at eight native-schedule steps with demonstrably adequate student capacity, run the same interval-gradient diagnostic before TDM training, and use the released FastH3 DMD2 checkpoint as the backbone-specific positive control. Do not begin H3 work before the Wan gate is evaluated.
- Re-read the complete handoff, root/modular-training instructions, and the `launch-experiment` skill. Its legacy experiment-journal step remains superseded by repository guidance; state stays in this active issue handoff. Preflight is clean except for this handoff: `gh` is authenticated as `macthecadillac`; issue 775 remains open/assigned with the same two comments; no open `775 OR TDM` PR exists; local/fork branch tip is handoff commit `27039de37982b6dd9626708c80f98682d2e6c1c1`; exact production-code commit remains `6676ef6b10ae240f0cb62dc8bf3d959faf777819`; and no issue-775 Kubernetes pod is active.
- The Wan ceiling is a temporary harness only. It starts the current synchronized attention-only rank-16/alpha-32 student LoRA from the pretrained Wan teacher function with fresh production AdamW state (`2e-6`, betas `[0,0.999]`, weight decay `0.01`, clip `1.0`). The critic is never called or stepped.
- Direct supervision is an exact flow-mapping upper bound. For every deterministic seed, the frozen Wan teacher generates a 50-step ODE endpoint with real encoded production-negative conditioning and CFG 6. For student source sigma `s`, the supervised input is the exact bridge `z_s=(1-s)x_teacher+s*epsilon` and the target is `x_teacher`. Exact prediction on all four production sigmas therefore makes the four-step ODE rollout follow that bridge to the paired teacher endpoint. This is deliberately a paired reachability diagnostic, not TDM or a distribution-matching objective.
- Rank `r` owns production interval/source sigma `r` for balanced global updates. Use four fixed training endpoints and four disjoint held-out endpoints per rank (16/16 globally), deterministic train seed base `710000` and held-out base `910000`, with full noise/target/conditioning fingerprints and no train-heldout overlap. Train 100 student-only updates, cycling the four training contexts, and evaluate steps `0/1/10/25/50/100`.
- Each evaluation records per-context direct target RMS/cosine on training and held-out bridge states, free four-step held-out rollout RMS/cosine to the paired 50-step teacher endpoint, local loss/gradient norm, represented LoRA parameter delta, and replica hashes. Assert exactly 100 student optimizer steps, zero critic parameter change, nonzero student change, and identical replicated student LoRA parameters.
- Predeclared positive gate: (1) final median direct-training RMS ratio to step 0 is `<=0.50` in all four intervals; (2) final median direct-heldout RMS ratio is `<=0.75` in at least 3/4 intervals including interval 3, with no interval above `1.05`; (3) across all 16 held-out seeds, median free-rollout RMS ratio is `<=0.70` and at least 12/16 contexts improve; and (4) all role/replica/finite/fingerprint invariants pass. Failure of any condition makes the four-step rank-16 Wan reachability gate negative.
- Run on one Kubernetes node with exactly four GB200 GPUs, exact code commit `6676ef6b1`, the retained overfit text-only data, PVC evidence, and immutable ConfigMap provenance. Preserve and checksum every JSON/input/log/status artifact before deleting only the completed pod. If the gate is positive, stop before H3 and return to critic conditional-expectation estimation. If negative, continue with the authorized H3 preparation; do not launch full H3 TDM before its interval-gradient gate.

### 2026-09-16 Wan student-reachability r1 ready to launch

- Completed the temporary harness and provenance runner as `tdm-student-reachability-6676ef6-r1`. The harness uses the exact production TDM flow conversions and Wan model wrappers, generates 50-step CFG-6 teacher endpoints, trains only the student LoRA in production training mode, leaves teacher/critic in evaluation mode, and records the predeclared direct/rollout gate at steps `0/1/10/25/50/100`.
- Removed redundant optimizer command-line overrides and instead assert the instantiated student optimizer exactly: learning rate `2e-6`, betas `[0.0,0.999]`, weight decay `0.01`, and max grad norm `1.0`. The server-side postprocessor independently rechecks optimizer values, schedule, seeds, context counts, unique/disjoint fingerprints, role deltas, replica synchronization, finite metrics, and gate shape.
- The pod manifest pins a single `NVIDIA-GB200` node and sets GPU request=limit exactly four. The runner clones enough branch history and detaches exact production-code commit `6676ef6b10ae240f0cb62dc8bf3d959faf777819`; it performs in-container AST validation before any model work and writes only to new PVC run root `/workspace/run/issue-775/tdm-student-reachability-6676ef6-r1`.
- Final input SHA-256: harness `26e0109a49c844a9fa07c873039bb49d33b6268bfe381e35d33da10f63eaa277`; runner `8d619ed6ad948fe2611dba4ae535ff18c1d2c9bcf876f838c8ad08a83946c5c7`; pod manifest `6cca0de28739d58852d9b2dbb69d8955c9a5bf7a9c2bfc5c2878cb69df0d69d2`.
- Kubernetes preflight eventually confirmed no issue-775 pod exists and the intended immutable ConfigMap name is unused, although the API experienced transient TLS-handshake timeouts and the first server dry-run timed out fetching OpenAPI. Retry the server dry-run before sealing provenance and submitting. Nothing has been created or allocated yet.
- The retry passed Kubernetes server validation. Created ConfigMap `tdm-student-reachability-6676ef6-r1-assets`, sealed it immutable at resource version `396835890`, and read every asset back from the API. Read-back SHA-256 values exactly match the harness, runner, and manifest hashes above. The pod has not yet been submitted.
- Submitted `vllm/tdm-student-reachability-6676ef6-r1`. Kubernetes scheduled it intact on node `10.0.130.11` with request=limit `4/4` NVIDIA GB200 GPUs. The container reports Torch `2.12.0+cu130`/CUDA 13.0 and four visible GB200s, detached exact code commit `6676ef6b1`, and passed in-container diagnostic AST validation. The distributed model/endpoint diagnostic is now running.

### 2026-09-17 Wan student-reachability gate completed negative

- r1 completed successfully at `2026-09-17T00:15:56Z` with `rc=0`. It deterministically generated all 16 training and 16 disjoint held-out 50-step CFG-6 teacher endpoints, completed 100 synchronized student-only optimizer steps, and produced the full predeclared report. All context/fingerprint, finite-value, role, and replica invariants passed.
- The gate is negative. Final median direct-training RMS ratios to step 0 by interval 0-3 were `0.927135 / 0.999270 / 1.000751 / 0.999708`, failing the required `<=0.50` in every interval. Held-out ratios were `0.931829 / 0.996791 / 0.998493 / 1.000497`; none met `<=0.75`, interval 3 did not qualify, and the required three intervals were absent. No held-out interval exceeded the `1.05` regression guard.
- All 16 free rollouts improved numerically, but only slightly: the median RMS ratio was `0.925407`, missing the required `<=0.70`. This distinguishes a small consistent update effect from practical reachability within the production 100-step optimizer budget.
- The replicated student LoRA changed by L2 `0.7961327931` on every rank and ended at the identical SHA-256 `448c89b53ed345911117ac0b428c6ca5c79bfbfa6e4a771129c89142ba95e0a9`. Every rank recorded exactly 100 student optimizer steps; critic optimizer steps and critic parameter delta were exactly zero.
- Interpretation: the current Wan rank-16 attention-only student does not reach the required four-step teacher mapping under direct paired supervision with the production AdamW settings and 100-update budget. This is an operational ceiling failure, not proof that no longer or retuned rank-16 optimization could ever fit. Per the authorized conditional plan, proceed to H3 preparation at eight native-schedule steps with full-rank or separately demonstrated adequate student capacity; run the interval-gradient gate before any H3 TDM training and retain the released FastH3 DMD2 checkpoint as the backbone-specific positive control.
- Preserved the complete 13-file run directory under `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-student-reachability-6676ef6-r1/`. Every file in the PVC artifact manifest matches locally. Key SHA-256: `report.json` `f556a0dd607c18d45b715e6386214b274c77a4c676cc96b8ccdd054c3f9abc99`; `summary.json` `cd985285db0abb81d71179698e7e60ddb7f68dfee79c979338d0ba797f6f00b6`; diagnostic log `7e68b3c65dfb0f327d82fac3378afb608fddb9f93e3873760cf21fab189a0d57`; orchestrator log `8e33bb82321e0e2797eb1816bde9118dc1bc979c5d51bd49fdefb3e4f5d1c480`; artifact manifest `45620733514c5dfe7fc098429ebd92b049211f1abfefb2f1ed504f3c0d6d3e21`. Retain the PVC run and immutable ConfigMap; delete only the completed pod before beginning H3 preparation.
- Deleted only completed pod `vllm/tdm-student-reachability-6676ef6-r1` after the checksum comparison. No issue-775 pod remains, so its four GPUs are released. The PVC run and immutable ConfigMap remain.

### 2026-09-17 H3 eight-step foundation preparation

- Rechecked GitHub as authenticated `macthecadillac`: issue 775 remains open/assigned with the same two comments and no open `775 OR TDM` PR. Upstream main is exact `9b0e57fe4b3a8c112ff24cc22f752eb0608ccfab`; official H3 DMD/VSA branch `feat/h3-dmd2-vsa` remains exact `08e18d333aba800fb469c9c14b5750f1469916d2`. No GitHub state was changed.
- The current issue code tip predates public H3 DMD support. Do not approximate the adapter with Wan's scalar-sigma math. The official H3 DMD branch supplies `MiniMaxH3DMDModel`, which packs joint video/audio latents, maps one base clock through modality-specific rational shifts, converts H3 noise-minus-clean predictions to x0 exactly, and exposes modality slices so audio is not overwhelmed by video's element count. Its README records successful full-weight three-role DMD2 execution on one node/four GB200 GPUs with SP=4 and HSDP shard=4.
- Use released positive control `FastVideo/FastVideo-FastH3-8-Step-V2`. Its official inference contract is eight rungs `[999,874,749,624,500,375,250,125]`, nine sigma-grid points/eight transformer forwards, guidance scale 1, video/audio shifts `10/3`, VSA sparsity `0.8`, and 64-token tiles. Its checkpoint metadata identifies a full BF16 70.10 GB student at DMD2 step 1300. A uniform nine-point grid is explicitly invalid for this checkpoint.
- H3 work will therefore use exact upstream release/inference code `9b0e57fe4` for checkpoint/conditioning preparation and exact validated DMD adapter commit `08e18d333` for the temporary training diagnostic, with every transplanted temporary file hashed. Production issue-branch code remains unchanged until the H3 gate establishes that a port is worthwhile.
- First H3 stage is asset preparation only: cache the minimum complete transformer/text-conditioning snapshots for base `MiniMaxAI/MiniMax-H3` and the released eight-step student, make manifest-limited training/conditioning views without modifying source snapshots, encode one fixed real H3 prompt through the actual Qwen3-VL conditioner, and write a one-row text-only parquet plus fingerprints. This stage performs no optimizer step and uses one node/exactly four GB200 GPUs.
- Planned gradient diagnostic after asset verification: use the released full-weight eight-step student as the backbone-specific positive-control arm; initialize a full-weight critic from the base teacher; build deterministic shared student trajectories at all eight trained rungs; train only the critic on balanced fixed TDM interval contexts; then measure per-modality critic error and actual-versus-sample-target-oracle full-parameter generator-gradient cosine on four predeclared held-out contexts per interval. H3 guidance remains conditional-only at 1.0. Exact video/audio shifted-sigma transitions and modality-balanced losses are required.
- Do not launch full H3 TDM unless the positive control establishes a positive final-interval median generator-to-oracle cosine with at least 3/4 positive held-out contexts, finite/nondegenerate gradient norms, and no severe early-interval regression. If critic learning is insufficient to interpret the gradient, extend only the critic-only diagnostic under a predeclared cap rather than treating an untrained critic as a backbone failure.

### 2026-09-17 H3 asset job ready to launch

- Prepared temporary asset-only job `tdm-h3-assets-9b0e57f-r1`. It pins released student revision `3da2ddfe1954d9cda4c05b643dc0f26007a655c5`, base H3 revision `42ed227ee7df40d41602854ae760620d6eb651fe`, and FastVideo main commit `9b0e57fe4b3a8c112ff24cc22f752eb0608ccfab`. The runner fetches and detaches the exact commit, and the preparation harness prefers the model's modular manifest.
- The job is update-free. It downloads only the student transformer and real text-conditioning components plus the base transformer, asserts the published eight-rung FastH3 inference contract, creates manifest-limited immutable-source views, encodes the fixed multimodal prompt through the actual H3 Qwen3-VL path, and writes a one-row text-only parquet with tensor/file fingerprints. The pod requests and limits exactly four `NVIDIA-GB200` GPUs on one node.
- Preflight passed as `macthecadillac`: issue 775 is unchanged and open/assigned; the open `775 OR TDM` PR search is empty; local and fork branch tips both equal `27039de37982b6dd9626708c80f98682d2e6c1c1`; the only worktree modification is this handoff; and no issue-775 pod is active. Kubernetes server validation passed after one transient OpenAPI TLS-handshake timeout.
- Input SHA-256: downloader `757063e8e6de0f386a1a5c5d1a9e74ffa2e6e166a3fc9a7811254cd1adf8b187`; preparation harness `1f8bad9f8e8d5085ab8b00575644c8cc8dc4b43e864980ba293d00491d5429a5`; prompt `d3f6a68b6be158d4cd4eb43c5cb69fe346b04397c7f63fb17cbb6cf05ee14f72`; runner `bc94435db10c16971ebfa64f6811446e45ea13dd85ada720be218b5978c29618`; pod manifest `40bc1af91d498ad98bbbebdb41ae503076bc6942a79e174ef129b83c660abc09`.
- Next: create and seal immutable ConfigMap `tdm-h3-assets-9b0e57f-r1-assets`, read back every input hash, submit and monitor the asset pod, copy/checksum all evidence locally, and delete only the completed pod. Then implement the H3 critic/gradient diagnostic against the verified views; do not launch full TDM training.

### 2026-09-17 H3 asset r1 infrastructure failure and r2 retry

- Created and sealed immutable ConfigMap `tdm-h3-assets-9b0e57f-r1-assets` at resource version `396888689`; every API-readback asset hash matched the predeclared values. Submitted r1, which scheduled on node `10.0.130.11` with exact GPU request/limit `4/4`, but exited in two seconds before cloning, downloading, model loading, or scientific work because `/workspace/issue-775/h3-cache` was not writable.
- Preserved r1's five exact inputs plus pod log/status under `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-h3-assets-9b0e57f-r1/`; retained its immutable ConfigMap and PVC run-root remnant. Pod log SHA-256 is `8f20f6d3189e2572b437c632eef68447eaabf99cd98bb0510f18c5c1c7dc8a5a`; status SHA-256 is `047d750c666584aa119a5f952ad84e825b79aff0096424f5ff409a1fcce1e09f`. Deleted only the failed pod after preservation.
- Prepared isolated r2 changing only the retry name and placing its Hugging Face cache plus persistent H3 asset views beneath the job's known-writable unique run root. Model revisions, exact FastVideo commit, prompt, preparation behavior, and published H3 contract remain unchanged. Kubernetes server dry-run passed and no issue-775 pod is active.
- r2 input SHA-256: downloader `757063e8e6de0f386a1a5c5d1a9e74ffa2e6e166a3fc9a7811254cd1adf8b187`; preparation harness `1f8bad9f8e8d5085ab8b00575644c8cc8dc4b43e864980ba293d00491d5429a5`; prompt `d3f6a68b6be158d4cd4eb43c5cb69fe346b04397c7f63fb17cbb6cf05ee14f72`; runner `d64acc83e46bd7d573755b74acb0a377125c5f85796ed5007ef3fc8d45a212e4`; pod manifest `e279955881fbd341d609eb5b5ef0e0647edc1afcd5fc7bc33ac23757dc68676a`.
- Next: create/seal/read back r2 immutable provenance, submit r2, preserve and verify all completed evidence, then continue to the H3 gradient gate only after asset validation.

### 2026-09-17 H3 asset r2 submitted

- Created and sealed immutable ConfigMap `tdm-h3-assets-9b0e57f-r2-assets` at resource version `396895118`; all five API-readback SHA-256 values exactly match the r2 preflight record. Submitted pod `vllm/tdm-h3-assets-9b0e57f-r2`.
- r2 scheduled on node `10.0.129.200` with exact GPU request/limit `4/4`. The container reports four NVIDIA GB200 GPUs (189471 MiB each), Torch `2.12.0+cu130`/CUDA 13.0, checked out detached exact FastVideo commit `9b0e57fe4`, and passed the in-container AST gate. Pinned asset download/preparation is running.

### 2026-09-17 H3 asset stage completed

- r2 completed successfully at `2026-09-17T00:54:38Z` with `rc=0`. It verified the official FastH3 eight-step contract `[999,874,749,624,500,375,250,125]`, eight forwards/nine grid points, video/audio shifts `10/3`, CFG 1, VSA sparsity `0.8`, and tile size 64. Both model snapshots match their pinned revisions.
- The released full-weight student transformer contains 14 shards totaling `70,099,582,760` bytes; the base H3 transformer contains 14 shards totaling `66,280,504,216` bytes. Both reduced views use `modular_model_index.json`. The real Qwen3-VL prompt embedding has shape `[49,5120]`, RMS `30.8606586`, SHA-256 `7a123329f301a51bccdbae87225520eb6b4c7cc209033eecfaa333eb7d5f34c4`; its genuine all-valid mask has shape `[1,49]` and SHA-256 `760e9fb3...`. The one-row parquet SHA-256 is `15edc5d3e474a5a35a890066a0cb6c2d18bb6cb3ec4feef2e0aba49b1d79809a`.
- Copied all 16 small reports/logs/status/provenance files to `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-h3-assets-9b0e57f-r2/`. Every file covered by the PVC manifest matches exactly; `.run_done` is also present and empty as expected. Key SHA-256: `assets.json` `e8707c2faa86fd60387ffe4ee881859476c93b9be687c0d1732ebab981bc1656`; `download.json` `b4fb3ba254a5c32be2af2236b6d9cfcf94787b54a66d99391217f692674e5dac`; prepare log `33c66e3b6b0009ff1452bc20b1eb8a67bd2688ad508d3ac1aae193d21c6de0b2`; orchestrator log `168851f95a188e3256eb763d9380a79f6d49f71f40d1a5d734c69091b85819bd`; artifact manifest `a7a4586f802232144972e85870b7e4babf82f498c6553d739cf76c36149d0279`.
- Retained the 190 GB pinned cache/asset views under `/workspace/run/issue-775/tdm-h3-assets-9b0e57f-r2/`, both immutable r1/r2 ConfigMaps, and r1's PVC remnant. Deleted only the completed r2 pod after evidence verification; no issue-775 pod remains and all four GPUs are released.
- Next: implement the temporary eight-interval H3 critic/gradient diagnostic against these verified views and exact official adapter commit `08e18d333aba800fb469c9c14b5750f1469916d2`. Use the released student as the backbone-specific positive control, a full-weight base critic, real conditioning, modality-balanced exact H3 objectives, and the predeclared final-interval alignment gate. Do not launch full TDM training.

### 2026-09-17 H3 critic/gradient diagnostic ready to launch

- Prepared temporary job `tdm-h3-gradient-08e18d3-r1`; no production branch code changes. It checks out exact official H3 DMD adapter commit `08e18d333aba800fb469c9c14b5750f1469916d2` and transplants only the exact production TDM math file from signed issue code commit `6676ef6b10ae240f0cb62dc8bf3d959faf777819` (SHA-256 `4b59cb7bf73361279dc83dfd7d2cb694759ff0007d477eed5ad0986386184f22`) for flow transitions, SNR, and paper pseudo-Huber.
- All three roles use validated dense FLASH_ATTN/FA4 to isolate the backbone from VSA. The student is the released full-weight 33B FastH3 checkpoint; teacher and full-weight critic use base H3. SP=4 and HSDP shard=4 place one joint audio/video document across exactly four GB200s. Student optimizer steps are trapped and forbidden.
- The signed released-checkpoint contract overrides the older adapter branch's base video shift: all roles use video/audio shifts `10/3`, schedule `[999,874,749,624,500,375,250,125]`, CFG 1 conditional-only, production-sized geometry `768x1344x124`, and exact deterministic H3 Euler/ODE trajectory replay. Interval targets are uniformly sampled as integer base-clock points in `[next trained rung, source rung)`, excluding terminal zero.
- Build four deterministic fixed critic-training trajectories (seed base `810000`) and four disjoint held-out trajectories (`910000`, stride `10003`). The critic's first 128 updates cover exactly 16 unique contexts per each of eight intervals using context seed base `1010000`; held-out/gradient contexts use `1210000`, interval stride `20011`, and context stride `1009`. The full-weight critic uses the validated `8e-6`, betas `[0,0.999]`, weight decay `0.01`, clip `1.0` optimizer. TDM SNR clip 5 and importance clip 10 are computed independently for video/audio, then equal-weighted.
- Critic sufficiency at step 128 requires held-out modality-balanced median RMS to improve by at least 10% in at least 6/8 intervals including interval 7, with no interval regressing more than 10%. If it fails, the same fixed critic contexts repeat once to the predeclared 256-step cap; no full TDM is launched. Gradient results are still recorded at the cap but are scientifically interpretable only if this critic gate passes.
- For four held-out contexts per interval, calculate actual TDM and corresponding sample-target-oracle gradients in the full student parameter space. Use equal-weight per-modality paper pseudo-Huber with `c=0.00054*sqrt(d_modality)` and no DMD normalization. Exact gradient cosine is recovered from three fresh backward norms (`actual`, `oracle`, `actual+oracle`) without storing a second 33B gradient buffer.
- The positive-control gradient gate requires: interval-7 median actual-to-oracle cosine positive; at least 3/4 interval-7 contexts positive; all gradient metrics finite; all actual/oracle norms nondegenerate; and no interval 0-6 median cosine below `-0.10`. A positive H3 result requires both critic sufficiency and this gradient gate. Otherwise do not launch full H3 TDM.
- Preflight passed: Kubernetes server dry-run; no active issue-775 pod; `gh` authenticated as `macthecadillac`; issue 775 unchanged/open/assigned with the same two comments; no open `775 OR TDM` PR; local/fork branch tips both `27039de37982b6dd9626708c80f98682d2e6c1c1`; only this handoff is modified. Input SHA-256: harness `dd7104b8b2fcfa69c7465866da2d22ee5b8cc49f9f82e90dee97f9e28e373abf`; TDM source `4b59cb7bf73361279dc83dfd7d2cb694759ff0007d477eed5ad0986386184f22`; config `91d82a2c4d83465b25cb0779348d4eb32711ee73d1325ba29cc95c19e0b038c4`; runner `68cc0849a50d92a875130c792d368ba5c66e0ed6a81c54c76322c3ea570d1d02`; pod `c330ed06ebf3e2c23cd6c15edd6b0fbbb04ac3850044208ac4cb11f785930c16`.

### 2026-09-17 H3 gradient r1 import failure and r2 retry

- Created/sealed immutable r1 ConfigMap at resource version `396927712`, with all five readback hashes matching, and submitted r1 on node `10.0.140.245` with exact GPU request/limit `4/4`. It checked out exact adapter commit, passed AST/config/asset gates, then stopped before model loading, trajectory generation, or optimizer work: importing the complete newer TDM method module required `synchronize_lora_gradients`, which the older official H3 adapter branch does not contain. The pod exited 1 at `2026-09-17T01:22:08Z`. Retain its ConfigMap and PVC logs/source/input-hash evidence.
- Prepared isolated r2 with no scientific-design changes. It retains the full exact signed TDM source for provenance and imports only a standalone helper containing the verbatim production implementations of sigma expansion, effective noise, flow SNR, transition-to-noisier-sigma, and paper pseudo-Huber. This removes incompatible LoRA/method scaffolding rather than approximating any equation.
- r2 server dry-run passes. Input SHA-256: harness `9c60711bc5fba22d5ec05e1e912efe52e348a5d8a53b1216fd4fd805f5c59b5e`; full TDM source `4b59cb7bf73361279dc83dfd7d2cb694759ff0007d477eed5ad0986386184f22`; extracted math helper `f48d796e033b46eec8f43522b10c2f29f3585b5fdf3eac39f9cac5292585c97e`; unchanged config `91d82a2c4d83465b25cb0779348d4eb32711ee73d1325ba29cc95c19e0b038c4`; runner `020b152d08f8c20af4e6e961e0061141ba7b27a82df5ad4f5a08cd77d7c29c8b`; pod `a0f297424cce56cc09519b2b20cfba41dc266d1b7d987c3c829eda985112dc3f`.
- Delete only failed r1 after this preservation record, create/seal/read back r2 provenance, and submit the unchanged exact-four-GB200 diagnostic.

### 2026-09-17 H3 gradient r2 checkpoint-load failure and r3 retry

- Deleted only failed r1 after retaining its immutable ConfigMap/PVC evidence. Created/sealed r2 ConfigMap at resource version `396934756`, verified all six readback hashes, and submitted r2 on node `10.0.140.245` with exact GPU request/limit `4/4`.
- r2 passed exact commit, AST/config, asset, distributed startup, and standalone TDM-helper import gates, then stopped while loading the first role. The older adapter branch's transformer class requires `transformer_blocks.29.attn.to_gate_compress.weight`, which the newer released FastH3 checkpoint does not contain. All four ranks failed identically. No role finished loading; no trajectory, loss, backward, or optimizer step occurred. The pod exited 1 at `2026-09-17T01:29:01Z`. Retain its ConfigMap and PVC evidence.
- Prepared r3 on checkpoint-compatible pinned FastVideo main `9b0e57fe4b3a8c112ff24cc22f752eb0608ccfab`, the same commit used successfully to verify the released assets. At runtime r3 fetches official adapter commit `08e18d333aba800fb469c9c14b5750f1469916d2` and transplants exactly its H3 training wrapper, H3 DMD adapter, and DMD2 algorithm layer while retaining current main's checkpoint-compatible transformer/loader implementation. This isolates adapter API support from transformer checkpoint schema. The config now targets the transplanted adapter module directly.
- Exact transplanted source SHA-256: H3 training wrapper `308dfa5cb9b7d8dccdeb423de04a64246717c0e1be4a49380134496d0744ad1b`; H3 DMD adapter `bfb3dc09349e652efc528d8aca329eabbef54940571d00c0f042fe3c6b3ee8cb`; DMD2 `d2af9f32c1678478df1cc639cd3251d57c98d0faa801a8eb7837040944ffd8af`.
- r3 server dry-run passes. Input SHA-256: harness `892b6f8f5b1f87ce7d865a2f197cea2f6371536e3ec6e8fa4f252b9aa5803ae5`; full TDM source `4b59cb7bf73361279dc83dfd7d2cb694759ff0007d477eed5ad0986386184f22`; extracted math helper `f48d796e033b46eec8f43522b10c2f29f3585b5fdf3eac39f9cac5292585c97e`; config `1e3012ec2f50e763b1796852964950a4fd006174096bb18db856c92f772cceca`; runner `a0d2ab3ff3f3fc9d9aa36e727647a6a8dbab23c88864a2ffc285fb6a2775632a`; pod `1d294a7043490a1939fc24e7289b8293a17adf3014c1f463a094026e8795a30f`. Scientific design/gates/seeds remain unchanged.

### 2026-09-17 H3 gradient r3 attention-backend startup failure

- Created/sealed r3 ConfigMap `tdm-h3-gradient-9b0e57f-r3-assets` at resource version `396941797`, verified every API readback hash, and submitted r3 on node `10.0.130.11` with exact GPU request/limit `4/4`.
- r3 passed pinned main/adapter checkout, AST/config/topology, asset-report, and distributed startup gates, then stopped while constructing the first role. The current development image contains FlashAttention 2 but not the `flash_attn.cute` module required when `FASTVIDEO_FA4=1`; all ranks failed before any role finished loading. No trajectory, loss, backward, or optimizer step occurred. This is an environment/backend-selection failure, not a scientific result.
- Prepare r4 changing only the forced FA4 environment setting: retain production `FLASH_ATTN` but let the installed backend resolve to the image's usable FlashAttention implementation. Keep the exact runtime/adapter/TDM sources, assets, seeds, geometry, optimizer, metrics, and predeclared gates unchanged. Preserve and verify r3 PVC evidence before deleting only its failed pod.
- Preserved the complete small-file PVC evidence for r1-r3 under `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/`. Every copied artifact matches its live PVC SHA-256 manifest; normalized manifest hashes are r1 `388df167d449e0640876dea387634dbc34d0966fd8799117cffe7fcfb0284fce`, r2 `eb6a055dd1a51f4cd5f5d2f32c729b3dcaea9f79c3e173bd6272d1572d7b624c`, and r3 `db5b96277c2df02009d0c183b2c81c1578107060833e940b71c8969e14caba36`. Immutable ConfigMap API snapshots are also local. Deleted only failed r3 and the zero-GPU evidence reader after verification; no issue-775 pod remained.
- r4 server dry-run passes. Input SHA-256: harness `0472cfbaab27bc521b37073b0f72bbb8dec7b237f89023f275986fd6893faec8`; full TDM source `4b59cb7bf73361279dc83dfd7d2cb694759ff0007d477eed5ad0986386184f22`; math helper `f48d796e033b46eec8f43522b10c2f29f3585b5fdf3eac39f9cac5292585c97e`; unchanged config `1e3012ec2f50e763b1796852964950a4fd006174096bb18db856c92f772cceca`; runner `fda7c99f815953d89c7b6fd382b340c53155f7928ab910e1150a58b297084470`; pod `87739895584f788502f7e60a409cca0810a7de990cba371da664a40f4e3fb854`.
- Created and sealed immutable ConfigMap `tdm-h3-gradient-9b0e57f-r4-assets` at resource version `396956479`; all six API readback hashes match the values above. Submitted r4 on node `10.0.129.200` with exact request/limit four NVIDIA GB200 GPUs. Startup reported exactly four visible GPUs, checked out pinned runtime `9b0e57f` and adapter `08e18d3`, and passed harness/config/topology and pinned-asset gates before entering the distributed diagnostic.

### 2026-09-17 H3 gradient r4 dense-checkpoint loader failure and r5 retry

- r4 resolved to the installed FlashAttention-2 backend and passed distributed startup, then loaded all 14 released student shards before the strict loader rejected `transformer_blocks.29.attn.to_gate_compress.weight`. No role finished construction; no trajectory, loss, backward, or optimizer step occurred. This is not a scientific result.
- The released FastH3 student index contains exactly 50 `transformer_blocks.{0..49}.attn.to_gate_compress.weight` entries; the base H3 index contains zero. Current FastVideo instantiates those learned compression gates only for the VSA backend. The deliberately dense `FLASH_ATTN` positive control therefore has no gate modules, and strict loading rejects checkpoint entries that dense execution cannot use.
- Preserved r4's complete small-file evidence under `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-h3-gradient-9b0e57f-r4/`; every copied file matches its live PVC SHA-256 manifest, whose normalized hash is `62f04eb16dac9b41e4a5424b14aa588b340fa72815235828faeaad1762a27fd1`. Immutable ConfigMap and pod API snapshots have SHA-256 `75604d9be33beda792d58eb13e5c6c2c62871a8941258f597e5b20007fe5bea7` and `954e9070a9155d4a76fd4ece1c953115be827be7e61ae4122fdf59f11acf723a`.
- Prepared r5 with one narrow runtime-only loader exception: use the loader's existing non-strict path for `MiniMaxH3Transformer3DModel`. Before loading it asserts the student/base gate-key sets are exactly 50/0. The harness asserts all three dense roles instantiate zero gate parameters. After completion the runner requires that every skipped checkpoint key is a gate weight and that the unique set is exactly all 50 expected student gate keys; any other unexpected or missing parameter still fails. Scientific inputs and gates remain unchanged.
- r5 server dry-run passes. Input SHA-256: harness `aa872227fa851fc554558eb16058e4d60c1ca45c1c8409333fdd0b015d72474c`; full TDM `4b59cb7bf73361279dc83dfd7d2cb694759ff0007d477eed5ad0986386184f22`; math helper `f48d796e033b46eec8f43522b10c2f29f3585b5fdf3eac39f9cac5292585c97e`; dense-loader patch `857844486722d7b9a444d9f2b900df43f4febf324eb652c2399c0f81f55f2dd9`; config `1e3012ec2f50e763b1796852964950a4fd006174096bb18db856c92f772cceca`; runner `af5aa3fc7734d72696e7d582516f536157fc6b2855647755a01d3b8662096a49`; pod `df984f14934e5e60ec99de5b102519f2c503a93898ba709fd60cc543f2766a2d`.

### 2026-09-17 H3 gradient r5 padding-fingerprint failure and r6 retry

- Created/sealed r5 ConfigMap at resource version `396973890`, verified all seven API payload hashes, and submitted r5 on node `10.0.129.200` with exact request/limit four GB200 GPUs. It passed runtime patch, asset, 50/0 gate-key, and distributed startup gates. It successfully loaded the released 33.12B student plus both 33.12B base roles; every logged unexpected source key is a student VSA gate weight. It then stopped before any trajectory or optimizer work because the raw-batch fingerprint compared the dataloader's padded embedding against the original 49-token asset.
- This assertion was invalid: the standard collator appends zero embedding rows and emits a float 0/1 mask, while `MiniMaxH3Model.prepare_batch` removes padding through that mask before casting the 49 real tokens to BF16. The asset report fingerprints the original unpadded FP32 `[49,5120]` embedding and a synthetic bool `[1,49]` all-valid mask. The mismatch is therefore representation, not conditioning corruption.
- Preserved r5's complete small-file evidence under `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-h3-gradient-9b0e57f-r5/`; every copied file matches its live PVC SHA-256 manifest, whose normalized hash is `3c1f962dff31d10eea22a7452fff47020d54f1a2ccba48e78c2343d9cfadf7c3`. ConfigMap/pod API snapshot hashes are `f4b6230ab2475677fa78f2b02f9b9f198cb9aa85c24963abbcb1054cdf78da1f` and `8583ca01dfed36611fa5fa7c5be49898ff8212fa0df91f1cadda7b9efba66da0`.
- Prepared r6 changing only conditioning validation: require a binary contiguous-prefix mask; require every padding value to be zero; fingerprint the exact unpadded FP32 tokens and equivalent bool valid mask against the asset report; retain padded fingerprints for evidence; and require `prepare_batch`'s BF16 conditioning to be bit-exact to an explicit mask-select-and-cast of those same tokens. Scientific design/seeds/gates remain unchanged.
- r6 server dry-run passes. Input SHA-256: harness `a071ca989b11d733e3b0683a0b52f44eb5696a0556d88f0bf3ed2dc62adedb73`; full TDM `4b59cb7bf73361279dc83dfd7d2cb694759ff0007d477eed5ad0986386184f22`; math helper `f48d796e033b46eec8f43522b10c2f29f3585b5fdf3eac39f9cac5292585c97e`; dense-loader patch `857844486722d7b9a444d9f2b900df43f4febf324eb652c2399c0f81f55f2dd9`; config `1e3012ec2f50e763b1796852964950a4fd006174096bb18db856c92f772cceca`; runner `cddac8e346c57037054720320e74dbade743c96fc8e2d49dac76db5f4ed32fa0`; pod `9d8a0d98dc00b3af7cdebda017e26ca25b7c759bf70e7bdbc6f77dd297ab11d8`.
- Created and sealed immutable r6 ConfigMap at resource version `396986685`; all seven API readback hashes match. Submitted r6 on node `10.0.130.11` with exact request/limit four GB200 GPUs. Startup reported exactly four visible GPUs and passed pinned checkout, harness/config/topology, asset, and 50/0 dense-loader checkpoint-contract gates.
- r6 loaded all three 33.12B dense roles, passed exact unpadded/prepared-conditioning validation, built the eight fixed trajectories, and completed the primary 128 critic steps with finite losses/gradients and stable memory. The held-out sufficiency gate at 128 did not pass, so the unchanged harness entered its predeclared extension to the hard cap of 256; step 144 was reached without failure. Await the capped held-out evaluation and gradient diagnostic before interpreting the H3 positive control.

### 2026-09-17 H3 critic/gradient positive-control completed negative

- r6 completed successfully with `rc=0` at `2026-09-17T03:17:36Z` on node `10.0.130.11`, one node, and exact GPU request/limit `4/4` NVIDIA GB200s. Runtime source was pinned FastVideo main `9b0e57fe4b3a8c112ff24cc22f752eb0608ccfab`, official H3 adapter source `08e18d333aba800fb469c9c14b5750f1469916d2`, and exact signed TDM math from `6676ef6b10ae240f0cb62dc8bf3d959faf777819`. No production branch code changed.
- Checkpoint/runtime contracts passed. The released student/base indices contained exactly `50/0` VSA gate keys; all three deliberately dense roles instantiated zero gate parameters; the only non-strictly skipped source keys were exactly the 50 expected student VSA gate weights (200 rank-local log entries, 50 unique keys). All three 33.12B roles loaded. The critic initially matched the teacher exactly (`max_abs_difference=0`), while the released student differed from the teacher (`RMS=0.020429695`).
- Conditioning validation passed exactly: the original unpadded FP32 embedding was `[49,5120]`, RMS `30.8606567`, SHA-256 `7a123329f301a51bccdbae87225520eb6b4c7cc209033eecfaa333eb7d5f34c4`; the real bool mask was `[1,49]`, SHA-256 `760e9fb3572cc48e4e9b2d7410133b40a071517df6936cc6966730a4bc89980d`. The padded representation had a binary contiguous-prefix mask, all `4,992,000` padded embedding values were zero, and `prepare_batch`'s BF16 tokens were bit-exact to explicit mask selection and casting.
- Update/isolation invariants passed: exactly 128 unique training and 32 held-out context fingerprints, no train/held-out context or trajectory overlap, exactly 256 critic optimizer steps, no student optimizer steps, student and teacher parameter hashes unchanged, and critic parameters changed. Source-prediction and target-context replay differences were zero for every gradient context.
- The full-weight critic failed the predeclared held-out fitting gate both at step 128 and at the hard cap of 256. Step-256 held-out modality-balanced RMS ratios (after/before) by interval were `[0.931972, 0.887446, 0.920338, 0.940345, 0.985484, 0.902244, 0.924897, 0.924587]`. Only interval 1 met the required 10% improvement; interval 7 improved only `7.54%`. No interval regressed by more than 10%, but the gate required at least 6/8 intervals including interval 7. For reference, step-128 ratios were `[0.937818, 0.882993, 0.885967, 0.896878, 0.938483, 0.910069, 0.927243, 0.939335]`.
- The gradient gate also failed. Median actual-to-sample-target-oracle cosine by interval was `[-0.368235, +0.051680, +0.259609, +0.002298, -0.080496, -0.065015, +0.331333, +0.615632]`, with positive-context counts `[2,2,2,2,2,2,4,4]`. All values and norms were finite/nondegenerate. Interval 7 was encouraging and passed its local criterion: all four contexts were positive at `[+0.839568,+0.620306,+0.610957,+0.144314]`. The global gate failed because interval 0's median was below `-0.10`; its four context cosines were `[-0.753399,+0.742311,+0.016930,-0.837965]`.
- Scientific decision: the released FastH3 student does remove Wan's specific final-interval sign failure under this diagnostic, but the backbone positive control is **negative/inconclusive overall** because the full-weight H3 critic did not clear the prerequisite held-out fitting gate, and the gradient gate also fails at interval 0. The gradient values must remain diagnostic rather than a valid backbone/TDM acceptance result. Do not launch full H3 TDM. The next experiment should extend or improve only H3 critic estimation (the unresolved norm-matched critic-weighting/update-direction control is the leading candidate), then rerun the same frozen-student interval diagnostic. Do not return to the completed Wan capacity ceiling.
- Preserved r6 evidence under `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-h3-gradient-9b0e57f-r6/`. All 27 manifest-listed local files match their PVC SHA-256 entries. Key SHA-256: `diagnostic.json` `fd4ea51e0f048d2bf8f165c97e4357ed5b9b8a43dbb7a9b746f41520fc97e83d`; `summary.json` `748b55ee9f1c0fde956cd750a07861f073f813b847296304358a3509b85da493`; diagnostic log `6176aaabf4200fc72e82149acf63f4350a3f739a63bea44d441d66d8929775cd`; orchestrator log `ad41a7fda2b639f28c50114766236f52b049404593137778edf033d207227358`; artifact manifest `7834f1fff35374649760175b2ce78f2e8aff481d84c550cd30692a0956e57549`; `.run_done` `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; ConfigMap API snapshot `399a43774fc63e7c61d9170c7b22c53d17faff8e21551b53c50c5df8329651be`; pod API snapshot `6b065d70c5b30807d5683e45a83da6850d0c974abe353591bf14ad09d203e349`. Retain the PVC run and immutable r6 ConfigMap (`resourceVersion 396986685`).
- Current repository rules supersede the earlier handoff-commit policy recorded above: this active handoff must remain local and must not receive a new commit/push. The branch and fork tip therefore remain `27039de37982b6dd9626708c80f98682d2e6c1c1`; no production or harness code is being committed from this temporary experiment.
- Cleanup: after all evidence and provenance hashes were verified and this handoff was updated, deleted only completed pod `vllm/tdm-h3-gradient-9b0e57f-r6`; `kubectl get pods -n vllm -l issue=775` now reports no resources, so all four GPUs are released. PVC evidence and immutable ConfigMaps remain.
- Final state check passed as authenticated `macthecadillac`: issue 775 remains open/assigned with the same two comments; no open `775 OR TDM` PR exists; local and fork branch tips both remain exact signed commit `27039de37982b6dd9626708c80f98682d2e6c1c1`; `git diff --check` passes; and this mandatory local handoff is the only worktree modification. No GitHub state was changed.
- Sent the required completion notification successfully through `/home/sandbox/notify` with title `TDM H3 diagnostic complete`.

### 2026-09-17 proposed H3 critic update-direction control (not launched)

- Next experiment should remain critic-only and use the released FastH3 eight-step student as a frozen positive control. Do not launch coupled H3 TDM until a critic recipe clears both held-out fitting and generator-direction gates.
- Reuse the exact r6 H3 assets, dense FlashAttention roles, native schedule `[999,874,749,624,500,375,250,125]`, conditioning, fixed trajectory/context seeds, full-weight base critic, and sample-target-oracle calculation. No student optimizer step and no production-code change.
- Phase A is a cheap paired update-direction screen, not another full training run. From identical critic/optimizer snapshots, compare production clipped-SNR-times-importance weighting against one H3-specific candidate: importance-only, equal video/audio weighting. Use the same four predeclared training contexts per interval and the disjoint r6 held-out bank. Do not repeat the redundant unit arm; Wan already showed it is effectively identical to importance-only.
- Apply the exact AdamW updates, then rescale the candidate's represented cumulative parameter delta to the production delta norm within `0.5%`; additionally require its median held-out critic-output RMS change to be within `0.75x-1.25x` of production so parameter-space matching is not mistaken for function-space matching. Record per-modality gradient allocation, raw/matched delta cosines, held-out critic RMS/update-to-target cosine, and generator-to-oracle cosine per context and interval.
- Phase-A gate: candidate held-out update-to-target cosine gain at least `+0.02` in at least 6/8 intervals including 0 and 7; lower held-out modality-balanced RMS than production in at least 6/8 with no interval above `1.005x`; interval-0 generator cosine gain at least `+0.20` with at least 3/4 contexts improved; interval 7 remains positive with at least 3/4 positive and median regression no worse than `0.10`; no interval 1-6 median generator-cosine regression worse than `0.10`.
- Only if Phase A passes, run a paired 128-update critic-only comparison from identical base snapshots and on identical context order. Evaluate at 0/32/64/128. The candidate must clear the original critic gate (after/before held-out RMS `<=0.90` in at least 6/8 intervals including 7, none `>1.10`) and the gradient gate, with interval 0 additionally becoming positive in at least 3/4 contexts. Stop after this gate; do not automatically launch coupled training.
- If Phase A fails, treat simple critic-loss reweighting as unlikely on both Wan and H3. The next mechanism should be target-estimation variance/bias: average multiple independent transition/proposal-noise targets for each fixed student state and test whether the critic and generator direction converge, rather than increasing capacity, update count, or CFG changes again.
- Operational requirements remain one Kubernetes node with exactly four GB200 GPUs, a temporary harness, immutable ConfigMap/PVC provenance, checksum-verified local evidence, pod deletion only after verification, no Modal/local tests, no PR, and no review loop. The current request is planning only; nothing has been submitted.
- Planning preflight passed as authenticated `macthecadillac`: issue 775 remains open/assigned with two unchanged comments, no open `775 OR TDM` PR exists, local branch tip remains `27039de37982b6dd9626708c80f98682d2e6c1c1`, and `git diff --check` passes. Only this active handoff is modified; no GitHub state changed.
- Sent the required planning-completion notification through `/home/sandbox/notify` with title `TDM next-step plan ready`.

### 2026-09-17 H3 critic weight-direction control ready to launch

- Implemented temporary job `tdm-h3-weight-direction-9b0e57f-r1`; no production branch code changed. It reuses the exact validated r6 runtime, adapter/TDM math, pinned H3 assets, real conditioning, dense FlashAttention roles, eight-rung 10/3 schedule, four training/held-out trajectories, context banks, and sample-target-oracle implementation.
- Phase A resets the full-weight critic and its AdamW/scheduler state identically before each arm, then runs exactly 32 updates (contexts 0-3 for every interval) for production clipped-SNR-times-importance versus importance-only. It records true video/audio component-gradient norms on the first context of every interval, held-out prediction error/output-update/update-to-target metrics, and all four generator-oracle contexts per interval.
- The candidate cumulative parameter delta is captured from the exact raw AdamW trajectory, rescaled against the production delta, and reapplied relative to the unchanged teacher until its represented BF16 delta norm is within `0.5%`. The harness also requires the predeclared held-out function-space output-update ratio `0.75x-1.25x`; raw/matched delta norms and cosines are preserved.
- Phase B is executable only if all Phase-A conditions pass. It restarts both arms from the base critic, trains to 128 updates on identical context order, evaluates at 0/32/64/128, norm-matches the final candidate, and applies the original critic/gradient gates plus the interval-0 `3/4` positive requirement. It stops without coupled TDM under every outcome.
- Input SHA-256: paired harness `b1f3e7cec7c1d4e647e2cb34dffc6aef2dc92b20495a811eb094f1dc922a3fe6`; validated r6 base harness `a071ca989b11d733e3b0683a0b52f44eb5696a0556d88f0bf3ed2dc62adedb73`; report validator `17c41d6726d11c26658d12202dafb14865734ea036b6925138cd322f8b86a565`; TDM source `4b59cb7bf73361279dc83dfd7d2cb694759ff0007d477eed5ad0986386184f22`; math helper `f48d796e033b46eec8f43522b10c2f29f3585b5fdf3eac39f9cac5292585c97e`; dense-loader patch `857844486722d7b9a444d9f2b900df43f4febf324eb652c2399c0f81f55f2dd9`; config `32a24826a970a8521dd619914dc4bf2d058f1214be6bcb2f8c04bbd9b509073f`; runner `852239d5aac15cd756f11225e5a2aa65535a23c43d4a317ffe41a70456e64098`; pod `483d782a7d14bdca90856f9b0f0b39394c71cf06262e6d5a550d2cc2ae5db120`.
- Static AST and shell syntax checks passed, and Kubernetes server-side dry-run passed. No model/test workload was run locally.
- Launch preflight: local/fork branch tips remain exact `27039de37982b6dd9626708c80f98682d2e6c1c1`; this handoff is the only issue-worktree modification; no issue-775 pod is active; no GitHub state changed. Earlier in this same work segment, `gh` was verified as `macthecadillac`, issue 775 was unchanged/open/assigned, and no open `775 OR TDM` PR existed.
- Next: create and seal immutable ConfigMap `tdm-h3-weight-direction-9b0e57f-r1-assets`, verify every API-readback hash, submit the exact-four-GB200 pod, monitor through the conditional gate, preserve/checksum all evidence, delete only the completed pod, and report the scientific decision.
- Created and sealed immutable ConfigMap `tdm-h3-weight-direction-9b0e57f-r1-assets` at resource version `397188309`. API readback confirmed `immutable: true` and all nine stored payload hashes exactly match the predeclared values above. Submit the pod next.
- Submitted pod `vllm/tdm-h3-weight-direction-9b0e57f-r1` at `2026-09-17T05:10:39Z`. It scheduled on node `10.0.128.163` with exact GPU request/limit `4/4`; the container reports exactly four NVIDIA GB200 GPUs. Pinned runtime/adapter checkout, static harness/config/topology gates, pinned asset report, and dense-loader 50/0 checkpoint-key contract all passed. Model loading/diagnostic execution is in progress.

### 2026-09-17 H3 weight-direction r1 harness-isolation failure

- r1 loaded all three 33.12B roles, then stopped before trajectory construction, loss, backward, or optimizer work. The invalid assertion compared teacher and critic parameter-shard hashes. The frozen teacher and trainable critic use different distributed parameter layouts, so those hashes need not match even though r6 already proved their initial functions match exactly. No scientific result was produced.
- Preserved all 17 live-PVC files plus ConfigMap/pod API snapshots under `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-h3-weight-direction-9b0e57f-r1/`; every copied PVC file passed SHA-256 verification. Key hashes: normalized PVC manifest `73b634153a69dc1d0c93bd50f5d32922edf1be3728aba84a1d0c0f79b0ab9733`; diagnostic log `682beb64e9882af607c5538577cd6918d93648ed58c45bcb97102c35dfa90261`; orchestrator log `35d77a142db0f0af700b2d58ab8dd453663d00cd1b5e6bd42e3e57e052ba2074`; status `a7c204b685c062921dd465d056f413641168beecbdd2554be52a550a1f9f613d`; ConfigMap snapshot `4c4c1fcfe55a2fb3c3761612b55669b305bf2bb8d40cbe29af429bdca3328e7c`; pod snapshot `4242d51b97a7f9b42ae646d924f6dba7d4116b686414fce73079479360bbb875`.
- Deleted only the verified failed r1 pod and zero-GPU evidence reader. No issue-775 pod remains and all four GPUs are released; r1 PVC and immutable ConfigMap evidence remain.
- Prepared isolated r2 with the exact scientific design unchanged. It captures the critic's own initial local BF16 shards, restores and byte-checks those shards before every arm, and expresses all raw/matched parameter deltas relative to that snapshot. The teacher remains only the unchanged function-space baseline/oracle; r2 explicitly requires the validated critic-teacher initial output max difference to be zero. This removes every cross-role parameter-layout assumption.
- r2 static AST/shell checks and Kubernetes server dry-run pass. Input SHA-256: harness `18358b953deb477b3d5bd1fa2b6152d7617d2cb06378f5be9177cdaf3f8fa3bd`; base harness `a071ca989b11d733e3b0683a0b52f44eb5696a0556d88f0bf3ed2dc62adedb73`; validator `8b63fc89a1ed8cd5f6e7cd7ed22fb91c1caa43620a5211b95b7aada7418d6897`; TDM source `4b59cb7bf73361279dc83dfd7d2cb694759ff0007d477eed5ad0986386184f22`; math helper `f48d796e033b46eec8f43522b10c2f29f3585b5fdf3eac39f9cac5292585c97e`; loader patch `857844486722d7b9a444d9f2b900df43f4febf324eb652c2399c0f81f55f2dd9`; config `01c1ed1df60ff6fca9a936f6b70225095a32e593ea1e31c9278f19aae0b738ed`; runner `55f22a36ff6912c04a742eca9427593d32c0e740cd301111f1f22fcb4642ef15`; pod `c8b55707c6485838db092d1d38f30642e4c30d27d60178756de0ea71b752beaf`.
- Next: seal/read back r2 immutable provenance and launch the exact-four-GB200 retry. No scientific input, gate, context, or optimizer setting changed.
- Created/sealed immutable r2 ConfigMap `tdm-h3-weight-direction-9b0e57f-r2-assets` at resource version `397212473`; API readback confirmed `immutable: true` and all nine hashes above exactly. Launch r2 next.
- Submitted r2 at `2026-09-17T05:32:14Z` on node `10.0.128.163` with exact request/limit `4/4` NVIDIA GB200 GPUs. Runtime/adapter checkout, syntax/topology, pinned asset, and dense-loader checkpoint-contract startup gates passed; model loading is in progress.
- r2 loaded all three 33.12B roles, captured the critic's own initial BF16 shards, restored them successfully, and passed the r1 failure point. The initial critic/teacher function check passed exactly, fixed trajectories/held-out baselines were constructed, and the production-weighted Phase-A arm entered its 32-update loop. Steps 1-16 are complete with no error/status marker; all four GPUs are at 100% utilization and about 166 GiB each. First-context modality-gradient allocation is strongly audio-dominated on intervals 1-6 (audio fractions `0.8813, 0.8140, 0.9648, 0.9447, 0.8474, 0.9283`), versus `0.3808` at interval 0 and `0.6586` at interval 7. Continue monitoring through production evaluation, candidate reset/update/evaluation, the predeclared conditional gate, and Phase B only if Phase A passes.
- The production arm completed all 32 updates and all 32 generator-gradient contexts without error. Production generator-to-oracle cosines by interval/context are: i0 `[0.461037, 0.863282, 0.089226, -0.876917]`; i1 `[-0.386909, 0.355158, 0.282395, -0.136769]`; i2 `[0.945632, -0.576434, 0.779649, 0.373127]`; i3 `[0.790641, -0.187951, -0.335282, 0.067250]`; i4 `[0.266480, 0.074629, -0.299978, 0.103125]`; i5 `[0.078420, 0.004729, 0.416454, -0.078976]`; i6 `[0.437761, 0.000747, 0.393628, 0.661313]`; i7 `[0.813344, 0.524814, 0.560418, 0.151877]`. Interval 7 is positive in all four contexts (median about `0.5426`); production interval 0 median is about `0.2751` but only three contexts are positive.
- r2 is now in the expected unlogged full-model integrity/reset section between arms: capture the full float32 production delta, hash/statistically fingerprint every critic shard (including float64 sums/square-sums), restore the initial critic/optimizer, and evaluate the candidate step-0 baseline. All four workers remain runnable at about 99% CPU, node memory is healthy (956 GiB total, about 459 GiB available, no swap), and no error/status file exists. The exact audit is the runtime bottleneck; updated Phase-A estimate is 90-120 minutes from the r2 start.

### 2026-09-17 H3 critic weight-direction Phase A completed: negative

- r2 completed successfully at `2026-09-17T06:42:30Z` with `rc=0`. Both 32-update arms ran from byte-verified identical critic shards and optimizer/scheduler state, used identical ordered contexts, and were compared only after matching the candidate's represented cumulative BF16 parameter-delta norm to production within `0.5%`. The candidate matched at ratio `0.9958653` (relative error `0.0041347`) and the global held-out function-space output-update ratio was `0.888654`, inside the required `[0.75,1.25]` band.
- Phase A nevertheless failed decisively; Phase B was correctly not executed. Per-interval results below are candidate/production held-out error RMS ratio, candidate-minus-production update-to-target cosine, production and candidate generator-to-oracle median cosine, candidate-minus-production generator cosine, number of contexts improved/positive, and median candidate/production output-update RMS ratio:

| interval | error ratio | update-cos gain | production gen-cos | candidate gen-cos | gen-cos gain | improved | positive | output ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.9451 | +0.0671 | +0.2751 | -0.3610 | -0.6361 | 2/4 | 1/4 | 1.0895 |
| 1 | 0.9637 | +0.0601 | +0.0728 | +0.0717 | -0.0011 | 2/4 | 2/4 | 1.1967 |
| 2 | 0.9174 | +0.1649 | +0.5764 | +0.2423 | -0.3341 | 1/4 | 2/4 | 1.0744 |
| 3 | 0.9743 | +0.1942 | -0.0604 | +0.2238 | +0.2841 | 2/4 | 3/4 | 0.9417 |
| 4 | 0.9125 | +0.0083 | +0.0889 | +0.0033 | -0.0855 | 1/4 | 2/4 | 0.7857 |
| 5 | 1.0202 | -0.1001 | +0.0416 | -0.2315 | -0.2730 | 2/4 | 2/4 | 0.6663 |
| 6 | 1.0461 | -0.2242 | +0.4157 | +0.5309 | +0.1152 | 3/4 | 4/4 | 0.5695 |
| 7 | 1.0371 | -0.2426 | +0.5426 | +0.7451 | +0.2025 | 3/4 | 4/4 | 0.4933 |

- Production/candidate per-context generator-to-oracle cosines, in context order 0-3, are: i0 production `[+0.4610,+0.8633,+0.0892,-0.8769]`, candidate `[-0.7051,+0.8836,-0.0169,-0.8206]`; i1 `[-0.3869,+0.3552,+0.2824,-0.1368]` vs `[-0.4059,+0.6339,+0.5075,-0.3641]`; i2 `[+0.9456,-0.5764,+0.7796,+0.3731]` vs `[+0.9575,-0.6794,+0.7348,-0.2502]`; i3 `[+0.7906,-0.1880,-0.3353,+0.0672]` vs `[+0.7753,+0.3000,+0.1476,-0.3372]`; i4 `[+0.2665,+0.0746,-0.3000,+0.1031]` vs `[+0.0263,-0.0196,-0.3737,+0.5386]`; i5 `[+0.0784,+0.0047,+0.4165,-0.0790]` vs `[+0.2526,+0.0528,-0.5157,-0.6842]`; i6 `[+0.4378,+0.0007,+0.3936,+0.6613]` vs `[+0.4940,+0.4789,+0.7054,+0.5679]`; i7 `[+0.8133,+0.5248,+0.5604,+0.1519]` vs `[+0.7544,+0.8954,+0.6534,+0.7358]`.
- Gate accounting: update-to-target cosine improved by at least `+0.02` only at intervals 0-3 (4/8, not 6/8) and regressed at interval 7; held-out error improved only in 5/8 and exceeded the allowed `1.005x` ceiling at intervals 5-7; interval 0 worsened by `-0.6361`, only 2/4 contexts improved, and only 1/4 was positive; intervals 2 and 5 had generator regressions greater than `0.10`. Interval 7 stayed healthy (4/4 positive and improved), but that cannot rescue the candidate. The raw candidate/production parameter-delta cosine was only `0.07819`; after BF16 norm matching it was `0.07467`, showing the two weighting rules drive almost orthogonal parameter updates even at equal norm.
- Integrity: critic and teacher initial function outputs matched exactly; the released student differed from the teacher (`RMS 0.02043`); student optimizer step count was zero; student and teacher parameter fingerprints were unchanged; training/held-out trajectories were disjoint. Schedule `[999,874,749,624,500,375,250,125]`, shifts `10/3`, CFG 1, training trajectories `810000,820003,830006,840009`, held-out trajectories `910000,920003,930006,940009`, training/held-out context bases `1010000/1210000`, interval stride `20011`, context stride `1009`, and trajectory stride `10003` all matched the predeclared design.
- Scientific decision: importance-only/equal-modality critic weighting is not a viable repair. This negative result now holds on H3 as well as Wan. Do not run Phase B, coupled H3 TDM, another capacity ceiling, or another simple loss-weight arm. The next high-value mechanism is target-estimation variance/bias: for each fixed student state/context, average multiple independent transition/proposal-noise targets and measure convergence of held-out critic target error and generator-to-oracle direction as target multiplicity rises.
- Complete 1.3 MiB evidence is preserved at `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-h3-weight-direction-9b0e57f-r2/`; every copied file matches its live PVC SHA-256. Key SHA-256: `diagnostic.json` `4236e95d42747b94baee06f8241ebf2a76d327ab952e1af764e68399da2a557e`; `summary.json` `77d53966c90448e968287002fdc0ca0f466dda1b8585b6c79513bddb96b03759`; diagnostic log `4e492fdf1a4d33bc3fa592b0d5b0453df5913a2ed802ff9d92dd51f62871957d`; orchestrator log `cfa9a15e8d12900adab3fb36d5feef527367529d0b44dcadacbba27dca68d804`; PVC artifact manifest `2d59e6f5730153631a7f62d96af024e368e3d2c70241dfeb225b97e22210c422`; ConfigMap API snapshot `e69af6e05301ac5b94a73842426d1807967eb75e8b4bd27b8d44f34d9ca885f4`; pod API snapshot `dd6c392afd2806795d3fb0e1fa3b7ae5ac70776075853a75f47944c9621e8228`.
- An r3 fallback that changed only temporary delta bookkeeping to bounded GPU chunks was prepared under `/tmp` while r2's exact CPU accounting was slow, but r2 completed successfully before any r3 ConfigMap or pod was created. Do not launch r3; r2 is the authoritative result.
- Next operational steps: delete only the completed r2 pod after this handoff update, retain both r1/r2 immutable ConfigMaps and PVC evidence, confirm no issue-775 pod remains, verify local/fork/GitHub state is unchanged, and send the mandatory completion notification.
- Final cleanup/state check: deleted only completed pod `vllm/tdm-h3-weight-direction-9b0e57f-r2` after the full local/PVC checksum comparison; `kubectl get pods -n vllm -l issue=775 -o name` is empty and all four GPUs are released. Retained immutable ConfigMaps r1 (`resourceVersion 397188309`) and r2 (`397212473`) plus their PVC evidence. `gh` is authenticated as `macthecadillac`; issue 775 remains open and assigned to `macthecadillac` with the same two comments; no open `775 OR TDM` PR exists. Local and fork branch tips both remain signed commit `27039de37982b6dd9626708c80f98682d2e6c1c1`; the only worktree modification is this mandatory local handoff, and no GitHub state was changed.
- Sent the mandatory completion notification through `/home/sandbox/notify` with title `TDM H3 weighting gate complete`.

### 2026-09-17 Source audit, critic-dominance finding, and authorized critic-convergence step

- User asked for a fresh code-level comparison of the current branch against the official TDM reference and paper, and a next-step recommendation. Re-read the full handoff, the production port `fastvideo/train/methods/distribution_matching/tdm.py`, the official reference `Luo-Yihong/TDM` at commit `81b019f91539948d49da86cef12ccc64030c5022` (`train_tdm_demo.py`), and paper arXiv `2503.06674` (Algorithm 1, Eqs. 6-12, Appendix D). Sent the required interim notification through `/home/sandbox/notify` with title `TDM review: critic fit is the blocker`.
- Source-audit verdict: the port objective is faithful. Flow transition `a=(1-s2)/(1-s1)`, `beta^2=s2^2-(a*s1)^2`, and `mixed_noise=(a*s1*eps_from+beta*proposal)/s2` match the reference `add_noise`/`obtain_mixed_noise` in the Wan flow parameterization; the importance ratio `exp(0.5*(proposal_sq-mixed_sq))`, `min(SNR,5)`, CFG-combined real target `real_cfg = real_uncond+(real_cond-real_uncond)*g` versus reference `(sd-fake)+(cfg-1)(sd-sd_uncond)`, the stop-gradient target `pred.detach()+(real_cfg-fake)`, denominator `mean|pred-real_cfg|`, paper Eq. 11 pseudo-Huber with `c=0.00054*sqrt(d)`, `separate` interval sampling, critic-before-generator ordering, and coupled trajectory reuse all match. The two deliberate deviations (actual effective noise versus the reference's predicted noise in the transition, and production negative-prompt/CFG semantics) were already validated as immaterial or non-causal. The reference's demo-time `mid`/`tau` direction and interval non-overlap also match the port.
- Recipe audit: the port's optimization budget is far below the paper's. Critic:generator LR ratio is `4x` (`8e-6`/`2e-6`) versus `10x` in the paper (`2e-5`/`2e-6` for SD1.5 and PixArt) and `5x` in the reference demo; effective batch is `4` versus `32` (PixArt), `64` (SDXL), `256` (SD1.5); the production config also uses `generator_update_interval: 5` where the paper updates the generator every iteration.
- New quantitative finding from the retained diagnostics (no new GPU work). Treating `actual = oracle + critic_error` in student-parameter-gradient space and using the three-backward identity `||err||^2 = ||actual||^2 + ||oracle||^2 - 2*||actual||*||oracle||*cos`, the critic-induced perturbation norm relative to the oracle signal norm is `1.0-2.2x` in all eight H3 r6 intervals (`2.2x` at interval 0) and `0.42-1.34x` in the Wan checkpoint-100 critic-gradient probe (`1.13x` at the failing interval 3). In latent space the H3 full-weight critic closed only `21-52%` of the base-teacher-to-student gap, and the same-context critic training loss moved only `1-11%` over its last 128 updates. Because a perfect critic makes the actual and oracle losses identical by construction, every negative generator-direction gate (Wan interval 3, H3 interval 0, and the norm-matched weighting comparisons) was error-dominated and measured critic fit, not the TDM objective. Capacity, CFG/conditioning, and weighting changes could not repair these gates because none of them fit the critic.
- Authorized next step: temporary job `tdm-h3-critic-convergence-<codecommit>-r1`, no production-code change. Freeze the released FastH3 eight-step full-weight student on the exact r6 assets/views, schedule `[999,874,749,624,500,375,250,125]`, shifts `10/3`, CFG 1 conditional-only, modality-balanced pseudo-Huber, and the r6 training/held-out context banks. Train only the full-weight base-H3 critic with AdamW lr `2e-5` plus a 50-update warmup, betas `[0,0.999]`, weight decay `0.01`, clip `1.0`, eight independent contexts per update (effective batch 32 across four ranks), balanced round-robin over the eight intervals, and unit weights (the production clipped-SNR-times-importance arm is retained only if runtime allows).
- Predeclared convergence gate: at the final evaluated step, held-out modality-balanced critic RMS ratio after/before must be `<=0.75` in at least 6/8 intervals, and the generator-gradient perturbation ratio `||actual-oracle||/||oracle||` must be `<=0.5` in at least 6/8 intervals including interval 0 with a positive median `cos(actual,oracle)`. Budget is 512 critic updates with evaluations at `0/64/128/256/384/512`, extensible once to a predeclared 1024 cap only if the curve is still improving and the gate is plausible. All invariants must hold: student never stepped, bit-exact source replay, finite metrics, replica-consistent roles, no train/held-out overlap.
- Interpretation: if the gate passes, the objective/plumbing is validated on a fitted critic and the next stage is the authorized paper-scaled joint H3 TDM run (critic `2e-5`, effective batch `>=32`, generator update interval 1, `>=1000` generator updates, distributional quality metrics). If the critic cannot converge even at the extended budget, the next investigation is the target/conditioning construction (modality balance, tau conditioning, source/target references), not another objective or capacity change. Run on one Kubernetes node with exactly four GB200 GPUs, retain immutable ConfigMap/PVC provenance, checksum all evidence, and delete only the completed pod.
- Preflight before the review: `gh` authenticated as `macthecadillac`; issue 775 open/assigned with its same two comments; no open `775 OR TDM` PR; local/fork branch tip `27039de37982b6dd9626708c80f98682d2e6c1c1`; no issue-775 Kubernetes pod active; worktree contains only this handoff modification. The handoff had accumulated uncommitted r1/r2 weight-direction entries from the prior segment and is being committed and pushed with this review record so no local state is lost on reboot.

### 2026-09-17 H3 critic-convergence job prepared (not yet submitted)

- User approved the plan and the session moved to build mode. The review record was committed as `9b0ef4d55` and pushed to `origin/issue/775-tdm`. This section corrects two provisional details in the earlier authorized-design entry: with SP=4 and HSDP shard=4, one forward pass across the four ranks processes one joint audio/video context, so context multiplicity is implemented by gradient accumulation before each optimizer step, and the final budget is 384 updates at 4 contexts per update (1536 context-gradients; 6x the r6 budget) at LR `2e-5` rather than 512/8.
- Prepared temporary job `tdm-h3-critic-convergence-9b0e57f-r1`, no production-code change. It reuses the exact r6 runtime (`9b0e57fe4`), adapter (`08e18d333`), dense FA2 loader patch, pinned H3 asset views, released full-weight student (frozen), base-H3 full-weight teacher and critic, eight-rung schedule `[999,874,749,624,500,375,250,125]`, shifts `10/3`, CFG 1 conditional-only, modality-balanced paper pseudo-Huber, and the exact r6 training (16/interval) and held-out (4/interval) context banks.
- Critic optimizer schedule: AdamW lr `2e-5` with a 50-update linear warmup then constant (the scheduler is not stepped; the harness sets the param-group LR explicitly), betas `[0,0.999]`, weight decay `0.01`, clip `1.0`; unit video/audio weighting so no interval is starved by clipped SNR (production SNR/importance weights are still recorded per context). The optimizer base LR and betas are asserted after method construction.
- Budget and evaluation: 384 updates, 4 contexts per update via `grad_accum_rounds=4`, balanced round-robin (`interval = step_index % 8`, `context_index = (step_index // 8) % 16`; the first 32 updates must cover exactly 128 unique training contexts with no held-out overlap). Critic held-out RMS evaluations at updates `0/64/128/192/256/320/384`; full generator-gradient decompositions at updates `0/192/384`.
- Metrics: per-interval held-out modality-balanced critic RMS after/before, the generator-gradient perturbation norm `||g_actual - g_oracle||` and ratio to the oracle norm via the three-backward identity, median `cos(g_actual, g_oracle)`, and latent-space `real-fake`, `real-student`, and `fake-student` RMS per context.
- Predeclared gate (unchanged from the authorized design): critic held-out modality-balanced RMS ratio `<=0.75` in at least 6/8 intervals and no interval above `1.0`; perturbation ratio `<=0.5` in at least 6/8 intervals including interval 0; interval-0 median gradient cosine positive; all gradient metrics finite/nondegenerate; student never stepped; bit-exact source/target replay; replica-consistent roles. If the critic converges but direction is still error-dominated, investigate the target/conditioning construction before any joint run; if the gate passes, the authorized next stage is the paper-scaled joint H3 TDM run.
- Input SHA-256: harness `c86f4d35a282e8e00cf16e18d4d22669f4a37c04c1c66e00eca3e8a9987260c6`; production TDM source `4b59cb7bf73361279dc83dfd7d2cb694759ff0007d477eed5ad0986386184f22`; math helper `f48d796e033b46eec8f43522b10c2f29f3585b5fdf3eac39f9cac5292585c97e`; dense-loader patch `857844486722d7b9a444d9f2b900df43f4febf324eb652c2399c0f81f55f2dd9`; config `494c2f9318a69cf106817083391123c637bba623a0e7ecce0795e5c94df84e94`; runner `66a808713e30b57470a8efc004f12b2a036bc5b6ac6f444c05cd886856ac47a5`; pod manifest `4c61b80b3099f5d849573520b1850568b9c7f708932916e2c3ae16e3a9c52fe0`. Local paths: `/tmp/issue-775-k8s/tdm-h3-critic-convergence-9b0e57f-r1/`.
- Static checks: Python AST parse passed for the harness and both transplanted sources; `bash -n` passed for the runner; the pod manifest passed Kubernetes server dry-run. Preflight: no issue-775 pod active, intended ConfigMap name unused, `gh` authenticated as `macthecadillac`, issue 775 open/assigned with no open `775 OR TDM` PR, worktree clean except this handoff. Next: create/seal immutable ConfigMap `tdm-h3-critic-convergence-9b0e57f-r1-assets`, read back every hash, submit, monitor, preserve/checksum evidence, delete only the completed pod, update this handoff, and send the completion notification.

### 2026-09-17 H3 critic-convergence gate completed: negative, with a fixed-bank confound

- Created and sealed immutable ConfigMap `tdm-h3-critic-convergence-9b0e57f-r1-assets` (resource version `397369385`); all seven API-readback payload hashes matched the prepared values exactly. Submitted pod `vllm/tdm-h3-critic-convergence-9b0e57f-r1`, which scheduled immediately on node `10.0.140.245` with exact GPU request/limit `4/4` NVIDIA GB200s. It checked out pinned runtime `9b0e57fe4` and adapter `08e18d333`, passed the dense-loader 50/0 gate-key contract, the pinned asset-report check, and the in-container harness/config/topology gate.
- r1 completed successfully at `2026-09-17T12:37:20Z` with `rc=0`. All invariants passed: student optimizer steps `0`, teacher and student parameter fingerprints unchanged, critic changed, `128` unique training contexts and `32` unique held-out contexts with no overlap, bit-exact source and target replay on every gradient context, finite/nondegenerate gradients, `384` critic updates with the exact `2e-5`/50-update linear warmup LR schedule asserted from the training records.
- Critic held-out modality-balanced RMS ratios (after over the teacher-initialized before) by interval at each evaluation update:
  - `64`: `0.743 0.783 0.905 0.902 0.943 0.947 0.973 0.981`
  - `128`: `0.644 0.760 0.837 0.915 0.940 0.921 0.959 0.978`
  - `192`: `0.615 0.752 0.834 0.881 0.963 0.922 0.955 0.973`
  - `256`: `0.610 0.760 0.854 0.914 0.994 0.901 0.941 0.964`
  - `320`: `0.651 0.763 0.851 0.899 0.985 0.897 0.941 0.964`
  - `384`: `0.671 0.777 0.851 0.896 0.978 0.885 0.934 0.957`
  Only interval 0 clears the predeclared `0.75` ratio, and every interval plateaus or slightly regresses after update ~128-192.
- The training fit keeps improving while the held-out fit stalls, which is the run's key confound. Mean video/audio training MSE by sixth of the run: for the noisier intervals 0-3 `0.1512 -> 0.0831 -> 0.0550 -> 0.0523 -> 0.0365 -> 0.0297`, and for intervals 4-7 `0.0131 -> 0.0111 -> 0.0106 -> 0.0105 -> 0.0101 -> 0.0099`; held-out MSE at step 384 is `0.165/0.118/0.050/0.018/0.018/0.010/0.016/0.017`. The full-weight 33B critic fits the fixed bank of 16 contexts from only 4 training trajectories per interval while failing to generalize to the disjoint held-out trajectories, so the held-out fitting gate conflates overfitting with any genuine target/conditioning defect.
- Generator-gradient decomposition (perturbation norm `||g_actual - g_oracle||` over oracle norm, and median `cos(g_actual, g_oracle)`, on held-out contexts):
  - step `0`: cosine `0.0` everywhere and perturbation ratio `~1.00`, exactly as expected because the critic equals the teacher so the actual signal is identically zero.
  - step `192`: cosines `0.452 0.251 0.173 -0.173 0.050 -0.391 0.553 0.695`; perturbation ratios `2.227 1.185 1.163 1.314 1.381 2.020 1.752 2.338`.
  - step `384`: cosines `0.350 0.289 0.223 -0.108 0.126 -0.163 0.587 0.611`; perturbation ratios `1.854 1.212 0.938 1.228 1.318 1.421 0.954 1.520`.
  Interval 0 was repaired relative to r6 (`-0.368 -> +0.350`), and six of eight intervals now have positive median direction, but no interval becomes signal-dominated: every perturbation ratio remains `>=0.94`, so the predeclared gate is negative.
- Latent-space critic progress at step 384: `||fake - student|| / ||teacher - student||` is `0.66 0.82 0.91 0.97 0.97 0.91 0.94 0.95` by interval. The critic removed only `3-34%` of the teacher's output distance to the student, and the low-noise intervals (2-7) have little headroom because the released student is already close to the base teacher there, making their per-context signal comparable to the residual posterior uncertainty of the student's x0 given the noised state.
- Scientific decision: the predeclared critic-convergence gate is negative, and per the predeclared branch no paper-scaled joint H3 TDM run is launched. The result is informative rather than a dead end: stronger critic optimization does improve the previously broken noisiest interval's direction, but the fixed 128-context bank means the held-out gate cannot distinguish generalization failure from an objective/conditioning defect, and the smallest-gap intervals have an intrinsic signal-to-noise limit in this frozen-student design.
- Recommended next step (not launched): rerun the same frozen-student critic fit with a materially larger training distribution and a train-context gradient measurement. Concretely: generate `>=32` training trajectories, sample fresh `tau`/proposal-noise contexts every update instead of cycling a fixed 128-context bank, keep the same held-out bank, and compute the generator-gradient decomposition on both training and held-out contexts. If train-context gradients become signal-dominated with a well-fit critic while held-out stays error-dominated, the blocker is context coverage; if train-context gradients are also error-dominated with a well-fit critic, the target/conditioning construction is genuinely suspect and should be investigated directly (modality balance, tau conditioning, source/target references).
- Complete evidence is preserved at `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-h3-critic-convergence-9b0e57f-r1/`; all 27 manifest-listed PVC files match their SHA-256 exactly. Key hashes: `diagnostic.json` `cae6c777fc6e8ea9a8379eeac5aa7faaa1e861d39ec5d11898b4294bfef52e2b`; `summary.json` `bab2831c01d9025378ce74cc8add37cf8edde66b7535babb6a3ddd598c7e276f`; `logs/diagnostic.log` `3c71f57106093493f28ffa2797a6c154d65070b78252619a197c5607bbd58ea0`; `logs/orchestrator.log` `8e7859109cb2e68780fe5b4c6f96005ac31f06a2d61dc32cf872f1e8728e96a7`; artifact manifest `9136cf1cac85e1785b84853b266ea94ee15218dfe5f79fee598da9854bcaacd6`; `status` `033087cdf7b795f2f4db83b23e648c5196f42181f5fa796d154ab0abc99ffba2`; ConfigMap API snapshot `193f8b068c1a6749ccd4c003a59a5ed095ca00463dcd006c0f16bbf2920f6c00`.
- Cleanup: retained the PVC run and immutable ConfigMap, deleted only the completed pod after the checksum comparison, and confirmed no issue-775 pod remains (four GPUs released). No production code changed and no GitHub state was changed.

### 2026-09-17 H3 critic context-coverage job prepared (not yet submitted)

- User approved the coverage experiment. Prepared temporary job `tdm-h3-critic-coverage-9b0e57f-r1`, no production-code change. It reuses the exact r6 runtime (`9b0e57fe4`), adapter (`08e18d333`), dense FA2 loader patch, pinned H3 asset views, released full-weight student (frozen), base-H3 full-weight teacher and critic, eight-rung schedule `[999,874,749,624,500,375,250,125]`, shifts `10/3`, CFG 1 conditional-only, and the modality-balanced paper pseudo-Huber.
- Context coverage change: the training pool is now `64` trajectories (`810000 + index*10003`), and every context draw uses a fresh hashed training trajectory (`splitmix64(step_index) % 64`), a deterministic round-robin interval, and a unique context seed (`2110000 + step_index*1009`), so every draw has fresh tau and proposal noise and there is no fixed context bank. Offline check of the hash stream: all `64` trajectories are used (`13-39` draws each) and each interval sees `60-63` distinct trajectories. The held-out bank is unchanged (4 trajectories `910000...`, 4 contexts per interval, seeds `1210000...`), and a new disjoint training-domain coverage bank uses 4 contexts per interval from the training pool with seeds `1610000...`.
- Critic schedule unchanged from the convergence run: AdamW lr `2e-5` with 50-update linear warmup then constant, betas `[0,0.999]`, weight decay `0.01`, clip `1.0`, unit video/audio weighting, `384` updates with `4` contexts per update via `grad_accum_rounds=4`.
- Evaluations: held-out critic RMS at updates `0/64/128/192/256/320/384`; coverage-bank critic RMS at `0/384`; per-context held-out gradient decomposition at `0/384`; and a new interval-aggregate gradient decomposition at `384` on both banks. The aggregate metric accumulates the four contexts per interval before measuring norms (matching the summed-batch student update) and recovers the dot product from three accumulated norms by linearity without holding two 33B gradients.
- Predeclared gate: the primary held-out gate is unchanged (critic RMS ratio `<=0.75` in >=6/8, per-context perturbation ratio `<=0.5` in >=6/8 including interval 0, interval-0 median cosine positive). The new coverage gate requires coverage-bank RMS ratio `<=0.75` in >=6/8, coverage aggregate perturbation ratio `<=0.5` in >=6/8, and positive coverage aggregate median cosine in >=6/8, with all metrics finite/nondegenerate. Interpretation: primary pass -> paper-scaled joint run; coverage pass with primary fail -> context coverage/generalization is the blocker (next test an online fresh-trajectory critic, not an objective change); both fail -> investigate the target/conditioning construction directly.
- Input SHA-256: harness `1e88e9d4654145a4adcf2927353b1084ab46b32138e9abb23bdb0ba7f39993da`; production TDM source `4b59cb7bf73361279dc83dfd7d2cb694759ff0007d477eed5ad0986386184f22`; math helper `f48d796e033b46eec8f43522b10c2f29f3585b5fdf3eac39f9cac5292585c97e`; dense-loader patch `857844486722d7b9a444d9f2b900df43f4febf324eb652c2399c0f81f55f2dd9`; config `ada18375cc2d81ac8abbf8d6d9ff4962d82d09f0096933b5d3c0c566bb9459c6`; runner `03f4ef9a01a34d296664b56cf90cf971ac6794b311fd988fc5127b0f7afda329`; pod manifest `3c28c5e5a92e6c919c1d9ef07bbe4089b04f6c69e4bef515a113e09bff62d90b`. Local paths: `/tmp/issue-775-k8s/tdm-h3-critic-coverage-9b0e57f-r1/`.
- Static checks: Python AST parse passed for the harness and both transplanted sources; `bash -n` passed; Kubernetes server dry-run passed. Preflight: no issue-775 pod active, intended ConfigMap name unused, `gh` authenticated as `macthecadillac`, issue 775 open/assigned with no open `775 OR TDM` PR, worktree clean. Next: create/seal immutable ConfigMap `tdm-h3-critic-coverage-9b0e57f-r1-assets`, read back every hash, submit, monitor, preserve/checksum evidence, delete only the completed pod, update this handoff, and send the completion notification.

### 2026-09-17 H3 critic context-coverage gate completed: negative; coverage hypothesis refuted

- Created and sealed immutable ConfigMap `tdm-h3-critic-coverage-9b0e57f-r1-assets` (resource version `398009502`); all seven API-readback payload hashes matched exactly. Submitted pod `vllm/tdm-h3-critic-coverage-9b0e57f-r1`, scheduled on node `10.0.140.245` with exact GPU request/limit `4/4` GB200s. It checked out pinned runtime `9b0e57fe4` and adapter `08e18d333`, passed the dense-loader 50/0 contract, pinned asset report, and harness/config/topology gates.
- r1 completed successfully at `2026-09-17T22:39:01Z` with `rc=0`. All invariants passed: student optimizer steps `0`, student/teacher parameter fingerprints unchanged, critic changed, `1536` unique fresh stream contexts, `64/64` training trajectories used, stream/held-out/coverage context banks disjoint, bit-exact source and target replay, finite/nondegenerate gradients, and the exact `2e-5`/50-update warmup schedule asserted from the training records. The 64-trajectory generation and all evaluations completed without harness errors.
- Both predeclared gates are negative. Held-out critic RMS ratios at `384`: `[0.474, 0.711, 0.833, 0.919, 1.005, 0.900, 0.934, 0.954]` (only interval 0 at `<=0.75`; interval 4 at `1.005` breaks the no-regression condition; per-context perturbation ratio `<=0.5` in `0/8`). Coverage-bank ratios: `[0.483, 0.756, 0.849, 0.896, 1.029, 0.934, 0.891, 0.964]` (only interval 0 at `<=0.75`; aggregate perturbation `<=0.5` in `0/8`; aggregate positive cosines `5/8`).
- Decisive comparison: coverage-bank fit equals held-out fit at every interval, and the training, coverage, and held-out MSEs agree per interval (intervals 4-7 all `~0.009-0.019`), so the critic is not overfitting the training trajectories or their `tau`/noise draws; the plateau is a domain-independent bias floor. The 64-trajectory fresh-context stream did not move the plateau: only intervals 0-2 improved relative to the fixed-bank run (absolute held-out RMS `0.287/0.315/0.218` versus `0.406/0.344/0.223`), while intervals 3-7 are unchanged.
- Gradient results at update `384`: per-context held-out cosines `[0.463, 0.406, 0.234, -0.046, 0.085, -0.274, 0.449, 0.649]` with perturbation ratios `[2.31, 1.17, 0.97, 1.25, 1.32, 1.54, 1.07, 1.21]`. Interval-aggregate held-out cosines `[0.574, 0.447, 0.804, 0.352, -0.215, -0.567, 0.607, 0.654]` with perturbation `[2.84, 0.92, 0.91, 0.94, 1.77, 2.05, 0.93, 1.36]`; coverage aggregate cosines `[0.119, 0.469, 0.240, 1.0, -0.238, -0.394, 0.546, -0.707]` with perturbation `[4.90, 0.89, 0.97, 0.69, 1.72, 1.59, 1.23, 2.61]`. Interval 0 remains the best interval (per-context `+0.463`, aggregate `+0.574`).
- Critic error relative to the teacher-student gap, `|fake-student| / |real-student|` = `0.46, 0.73, 0.87, 1.00, 1.01, 0.93, 0.94, 0.94`. In intervals 3-7 the released student is already close to the base teacher and the critic residual (posterior uncertainty of the student's source x0 given the tau-noised state) is comparable to the gap, so no fake score can make the generator gradient signal-dominated there. Intervals 0-2 have large gaps and improved substantially (interval 0 absolute RMS `0.605 -> 0.287`).
- Transient instability worth recording: update 47 had mean loss `4.43` with pre-clip gradient norm `384` (run median `0.41`); updates 33-64 mean train MSE was `0.213` versus `0.113` in updates 9-32 and `0.079` in 65-128, and the step-64 held-out evaluation was inflated `2-6x` before recovering by step 128. Fresh-context gradient variance is much higher than the fixed bank (max pre-clip norm `384` versus `19.9`). With effective batch `4` and full-weight AdamW at `2e-5`, single-context spikes can transiently damage the critic; the paper's batch `32-256` would average them out.
- Interpretation: the coverage hypothesis is refuted, and the predeclared "investigate target/conditioning construction" branch is not supported by this evidence. The remaining failure mode is structural to the frozen-student design: at low noise the student-teacher gap is at or below the critic's irreducible residual. Recommended next step (not launched): either (a) run the paper-scaled joint H3 TDM run where the student moves and the gap grows, with a larger effective batch to absorb gradient spikes, or (b) build a positive control with a deliberately larger student-teacher gap at all intervals (for example a partially trained or rank-limited student) so the gate is testable. Do not spend more compute on frozen-student critic schedules, coverage, or capacity.
- Complete evidence is preserved at `/home/sandbox/FastVideo/outputs/issue-775-tdm/k8s/tdm-h3-critic-coverage-9b0e57f-r1/`; all 27 manifest-listed PVC files match their SHA-256 exactly. Key hashes: `diagnostic.json` `7afdad4803a973e146fb826eaadce840d10dfa27e30de22cccf0dc1c868af643`; `summary.json` `2c23599acbf1415e697c7552c183583fe7ce2a94db6d8252e512464d302e60ae`; `logs/diagnostic.log` `2f251df4a317f910e090c25c2135dd37664003e97faf157a4cc2171db8c23ce8`; `logs/orchestrator.log` `7ad00d6cec7c74911fe0ccc200fdb40b135a296c6ca36480699bf2422b0296d0`; artifact manifest `3b0679aa57038d7c14416ba8f253313cb6b39e76d0fd030003a3a4b930c73742`; `status` `11f3e363e02fe5cb428e902fbda086b53e8cf3ff6d71e6c0f709c32fdcd2bb6e`; ConfigMap API snapshot `b4aac6a659c2b9004f3012abc4a3b812411ed0d86eadd4a0857f1d8d0590a563`; pod API snapshot `0ea5867757ef0fed24acbd011efb13fc38d962bcfc1378a4896b834def791be7`.
- Cleanup: retained the PVC run and immutable ConfigMap, deleted only the completed pod after the checksum comparison, and confirmed no issue-775 pod remains (four GPUs released). The `/tmp/fastvideo-worktrees/issue-775-tdm` worktree was pruned by environment cleanup during the run; it was recreated from the pushed branch at `28227592b` and this record is committed from there. No production code changed and no GitHub state was changed.

### 2026-09-17 Local generated-video review bundle

- User asked for the generated test videos alongside the teacher output for review. Assembled `/home/sandbox/FastVideo/tdm-review-videos/` (untracked local artifact, not committed): 44 files, `25 MB`, with `README.md` and a `manifest.sha256`; all 32 MP4s pass a container-header check.
- Contents: (1) `4prompts-20260714/` from `tdm-bsz4-500-k8s-20260712144347` (earlier TDM commit `dbcd9ac1`), with four prompts (`woman_earrings`, `horse_rider`, `poison_bottle`, `forest_river`), each with `teacher_50step`, `student_base_4step`, and `student_tdm_ckpt{100,300,500}`, plus midpoint montages and `quality_metrics.json` (ckpt100-vs-ckpt500 MS-SSIM `0.979`, ckpt500-vs-base `0.559`, ckpt500-vs-teacher `0.323`, base-vs-teacher `0.449`); (2) `1prompt-current-20260916/` from `tdm-coupled-pseudohuber-100step-r1` (current production code `6676ef6b1`, single "red toy car" prompt, baseline plus checkpoints `25/50/75/100`, contact sheet, comparison JSON); (3) `1prompt-postfix-20260915/` from `tdm-fixed-d1e4f676` (post-fix overfit, same prompt and checkpoints, contact sheet).
- Only rank 0 of each replicated inference is copied; the four ranks were previously verified pixel-identical. The recent H3 diagnostics produced no sampled videos, so the bundle covers the Wan overfit evidence only. The failure signature to review visually is unchanged: the student checkpoints remain a dark, blurred, mode-averaged scene and do not approach the sharp teacher.

### 2026-09-17 resume reconciliation after independent code review

- Read this handoff completely and reconciled it against the live worktree, fork, GitHub issue/PR state, retained coverage report, and Kubernetes state. The worktree is clean on `issue/775-tdm`; local and fork tips both equal signed handoff-only commit `575e2e6b3c8a20d8f5d29df634e6c2874eb4c117`. The production-code tip remains signed commit `6676ef6b10ae240f0cb62dc8bf3d959faf777819`; every later branch change is confined to this active handoff.
- `gh` is authenticated as `macthecadillac`. Issue 775 remains open and assigned with the same two comments, no open `775 OR TDM` PR exists, and no GitHub state was changed. `kubectl get pods -n vllm -l issue=775` reports no resources, so no experiment is running and no issue-775 GPU allocation remains.
- Exact stopping point: the fresh source audit found the implemented TDM objective faithful; the H3 critic-convergence and 64-trajectory fresh-context coverage experiments both failed their gates; and the coverage result refuted fixed-bank overfitting because training-domain coverage and disjoint held-out fit reached the same interval-wise floor. Do not spend more frozen-student compute on critic schedules, coverage, capacity, simple weighting, or conditioning controls.
- Bookkeeping correction: the final held-out critic RMS ratios were `[0.474, 0.711, 0.833, 0.919, 1.005, 0.900, 0.934, 0.954]`, so intervals 0 and 1 (2/8), not only interval 0, satisfy `<=0.75`. This does not change the negative gate, which required at least 6/8 and also failed the no-regression and gradient-perturbation conditions. The coverage-bank count remains 1/8 because its interval-1 ratio was `0.756`.
- Recommended next action remains the paper-scaled joint H3 TDM experiment where the student moves and the teacher-student gap can grow: critic LR `2e-5`, effective batch at least 32, generator update interval 1, and at least 1000 generator updates, judged with multi-prompt/seed distributional video-quality metrics rather than paired same-seed MS-SSIM. If a cheaper positive control is required first, deliberately enlarge the student-teacher gap at every interval with a partially trained or rank-limited student. No new run was authorized or launched during this reconciliation.

## Joint H3 TDM plan (revised 2026-09-21, grounded on origin/main)

Per user direction the plan is built on `origin/main` (`61b91220c`,
mirroring `hao-ai-lab/FastVideo:main`) rather than the experimental H3
branches. What main provides today:

- `MiniMaxH3Model` (`fastvideo/train/models/minimax_h3/minimax_h3.py`)
  with joint video+audio packing, per-modality scheduler shifts 12/3,
  per-row timesteps, `backward` restoring the forward context, and
  `predict_noise` returning `(noise - clean)` per modality. Its gates:
  `TORCH_SDPA` only, batch 1, cfg rate 0, `t2va` data, dense attention
  only, no unconditional/CFG path. It has no `predict_x0` override
  (the base raises on tuple returns), no LoRA, and its H3 scheduler
  exposes no `num_train_timesteps`.
- One H3 recipe: `examples/train/configs/overfit_minimax_h3_t2va.yaml`
  (fine-tune, 64 GPUs, sp 8 / hsdp 8x8, 768x1344x124,
  `data/crush-smol_h3_t2va_single_sample_preprocessed`) plus the two
  launch scripts next to it.
- Wan VSA training recipes and the H3 VSA inference backend, so VSA
  exists on main but is not wired into H3 training.
- `TDMMethod` on this branch already subclasses main's `DMD2Method`, so
  no cross-branch merge is needed: everything lands on main plus this
  branch's TDM code.

The experimental H3 DMD branches (`h3-dmd-*`) are consulted for lessons
and working values only - VSA90 tile 64, LoRA rank 64, shifts 12/3,
4-step `[999, 749, 500, 250]`, prompt-only data, LoRA
replica/gradient-sync pitfalls. No code depends on them.

### Phase 0 - adapter spike (1 day, no GPU)

- Decide the adapter shape: subclass `MiniMaxH3Model` as an
  H3 TDM model on this branch adding
  (a) per-modality `predict_x0` (`x0 = noisy - sigma * pred` with the
  model's own video/audio sigmas, verified against the standalone
  reference on `feat/tdm-standalone-h3-port`),
  (b) `num_train_timesteps = 1000` plus a sigma/timestep mapping for
  TDM's interval sampling (video shift 12, audio shift 3),
  (c) LoRA enablement through main's `_enable_lora_if_configured`,
  (d) CFG handling for a guidance-distilled model (guidance 1, no
  unconditional branch).
- Decide TDM's bundle interface: recommended is a modality-aware
  trajectory (one interval draw, per-modality sigmas, per-modality
  losses summed, as in the standalone `JointRollout`), rather than
  forcing everything through one packed tensor.

### Phase 1 - method and adapter (2-3 days, no GPU)

- Implement the H3 TDM model adapter on this branch; wire LoRA (rank 16
  to match the proven standalone recipe, rank 64 as the branch-lesson
  ablation) and the LoRA gradient synchronization already added on this
  branch (`utils/lora.py`, `synchronize_lora_gradients`).
- Make `TDMMethod` modality-aware; port the joint-rollout semantics and
  per-modality flow math from the standalone reference.
- Keep dense attention for the first gate; VSA comes later (Phase 5).
- Tests: extend `tests/local_tests/tdm/` with the H3 bundle math
  (per-modality x0 round-trip, sigma mapping, shared interval) against a
  CPU stub model.

### Phase 2 - config and data (1 day)

- Clone `overfit_minimax_h3_t2va.yaml`, swap the method for TDM, LoRA
  student+critic with a frozen teacher, `tdm_denoising_steps
  [999, 749, 500, 250]`, `generator_update_interval: 1`, critic LR
  `2e-5`, `gradient_accumulation_steps: 8` on four GPUs (effective batch
  32), geometry 768x1344x124, `max_train_steps >= 1000`.
- Cheaper first gate: the standalone's 480x832x124 geometry on the
  single-prompt overfit set before the production-shaped run.

### Phase 3 - acceptance metrics (1-2 days)

- Port the teacher-cloud report from the standalone
  (`tdm_standalone/tdm/overfit_report.py`: teacher leave-one-out
  nn-rel-MSE, split-MMD floor, student nn-rel-MSE, MMD, paired drift)
  as a post-hoc script over training checkpoints plus
  `MiniMaxH3Pipeline`.
- Pre-register: baseline and final student clouds inside
  `nn_rel_mse < teacher loo` and `mmd < 2x split_mmd` on at least four
  prompts x 32 seeds; paired same-seed MS-SSIM as a diagnostic only.

### Phase 4 - one-node/four-GPU run (1-2 days wall clock)

- Pod per the branch manifests (one node, four GB200;
  `/opt/venv/bin/python`, `TRITON_CACHE_DIR`, PVC paths).
- Preflight: targeted TDM tests, four-rank NCCL, LoRA replica-init
  fingerprint and gradient-sync check, two-step smoke with finite critic
  and student grad norms.
- Main run: at least 1000 generator updates, checkpoints every 100,
  resume-safe, JSONL metrics; then the distribution report and contact
  sheet. Evidence checksummed; delete the pod only after verification.

### Phase 5 - VSA follow-up

- Wire main's H3 VSA backend into the H3 train adapter using the
  standalone's validated recipe (tile 64 Triton, sparsity 0.5/0.9,
  student-only versus all roles), then rerun the distribution gate.
  Main already carries H3 VSA for inference and Wan VSA training
  examples, so this is a training-path wiring task, not new kernels.

### Risks

- The H3 adapter is new code (roughly 200-400 lines) rather than a
  merge; the standalone adapter is the math reference and the
  experimental branches are the hardware lesson.
- Memory: full-weight student plus critic plus AdamW does not fit on
  four GB200s; LoRA is mandatory.
- Queue contention for one-node/four-GPU allocations.
- The critic-gap question stays open until the student moves; the run is
  designed to answer it.

Effort: about one week to the pre-registered run, plus 1-2 days of
GPU wall clock.
