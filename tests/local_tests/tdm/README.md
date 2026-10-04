# TDM Local Tests

Local-only tests for the Trajectory Distribution Matching implementation under
`fastvideo.train.methods.distribution_matching.tdm`.

This is a Wan flow-matching adaptation of the original CogVideoX/diffusion TDM
reference, not a checkpoint-compatible port of the reference training script.
The tests in this directory focus on the math bridge and method wiring; the
method's behavior is documented in `docs/training/train_infra.md` (TDM and
TDM-on-MiniMax-H3 sections). GPU validation runs on Slurm: one node with
exactly four GPUs, no multi-node jobs.

## Reference Assets

| Field | Value |
|---|---|
| Model family | `wan` |
| Workload type | T2V training/distillation |
| Method | Trajectory Distribution Matching |
| Official reference | `https://github.com/Luo-Yihong/TDM` |
| Official reference file | `train_tdm_demo.py` |
| Original target | CogVideoX-2B diffusion training |
| FastVideo target | Wan 2.1 T2V 1.3B flow-matching LoRA training |
| Example config | `examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml` |
| Local reference dir | optional env `TDM_REF_DIR` |
| Wan weights | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` or local env `TDM_WAN_MODEL_DIR` |
| Runtime reference imports | none in production code |

Use only env-var names for tokens, such as `HF_TOKEN`. Never paste token
values into this file.

## Reference-To-FastVideo Mapping

| Reference concept | FastVideo implementation |
|---|---|
| `generate_new(...)` | `TDMMethod._student_trajectory(...)` |
| `Predictor.predict(...)` | `ModelBase.predict_x0(...)` using Wan scheduler output conversion |
| `Predictor.add_noise(...)` | `flow_transition_to_noisier_sigma(...)` for fake-score noising |
| `Predictor.obtain_mixed_noise(...)` | `flow_transition_to_noisier_sigma(...)` returned `mixed_noise` |
| fake-score update | `TDMMethod._tdm_fake_score_loss(...)` |
| generator update | `TDMMethod._tdm_generator_loss(...)` |

Each managed training step applies the fake-score optimizer update first. It
then generates a fresh student trajectory and evaluates the generator loss with
the updated critic. The shipped Wan configs encode the model's negative prompt
for the teacher's unconditional classifier-free-guidance branch.

Wan uses:

```text
x_sigma = (1 - sigma) * x0 + sigma * eps
x0_hat = x_sigma - sigma * model_output
```

The diffusion alpha-bar math from the reference is intentionally not used in
production code.

The generator update preserves the reference's three noise levels for each
sample: the student predicts at the source sigma, its effective flow noise
reconstructs a cleaner intermediate, and fresh noise moves that reconstruction
to a score target. The source is always sampled from the generated trajectory;
`use_randmid` randomizes the cleaner intermediate point. The target bounds are
`intermediate <= target < source` in `separate` mode and
`intermediate <= target < terminal` in `to_terminal` mode. Sampling is
independent per batch element.

Periodic validation loads a separate inference pipeline while the student,
teacher, and critic exist. The shipped recipe therefore offloads training state
during validation and unloads the inference pipeline afterward. Checkpoint-only
training can disable periodic validation with
`callbacks.validation.every_steps=0` and validate exported checkpoints in a
separate inference job.

## Warmup And Step Ladder

For Wan-family students the method defaults to a **200-update regression
warmup** (`method.warmup_steps`, set explicitly in the example config). During
warmup the student's x0 at its own rollout states is regressed onto the
guidance-combined teacher x0 at those same states; the critic is not updated,
is excluded from `get_optimizers` / `get_lr_schedulers` / grad-clip targets,
and its optimizer state stays empty. Other families default to no warmup
(guidance-distilled bases already start inside the teacher's basin).

The reason is empirical, from the standalone TDM experiments: on Wan, TDM
alone drifts out of the teacher distribution and warmup alone collapses
visually (blurry blobs on every seed), while warmup followed by TDM produces
coherent, sharp 4-step samples. The warmup is what puts the student's states
on-manifold so the critic correction stops dominating the generator signal.

A **progressive step ladder** is available through `method.tdm_step_ladder`
(commented example in the shipped config). Stage step counts must strictly
decrease (for example 8 steps then 4), only the final stage may omit
`until_iteration`, and validation/inference adopt the final stage's schedule
because the loader writes `dmd_denoising_steps` from it.

Learning rates matter as much as the phase split. The DMD2-inherited student
LR of `2e-6` is a no-op regime for TDM: the Phase 3 diagnostic measured only
about `1e-4` of adapter movement over 200 updates, and the treatment and the
regression-only control produced visually identical blurred samples. At
student `1e-4` with fake-score `1e-4` the warmup bakes guidance in and TDM
then sharpens to a coherent sample while the control stays blurred, so the
validated diagnostic configs (`tdm_t2v_lora_fixed_recipe.yaml`,
`tdm_t2v_lora_overfit.yaml`) ship those values. The production-shaped
`tdm_t2v_lora.yaml` keeps the DMD2 value with a warning comment because it has
not been re-validated at the higher rate.

Two measurement caveats carried over from the standalone work: paired
same-noise metrics (latent nearest-neighbour / MMD) and paired pixel MS-SSIM
rank blurry no-guidance samples above coherent distilled ones, so they must
not be used as quality gates for TDM; use reference-free signals (visual
coherence, frame sharpness, sample diversity, cross-prompt behaviour) and keep
the cloud metrics as diagnostics only.

## Acceptance And Diagnostics

| Signal | Implementation | Guidance |
|---|---|---|
| Frame sharpness (mean squared luminance gradient) | `tdm_metrics.frame_statistics` | The blur detector: a no-guidance or regression-only student falls far below the teacher; a distilled student sits in the teacher's ballpark or above |
| Frame contrast (std) | `tdm_metrics.frame_statistics` | Within roughly 10 percent of the teacher's |
| Latent-cloud diversity | `tdm_metrics.latent_cloud_statistics` | Student median pairwise distance within roughly 10 percent of the teacher's; mean off-diagonal cosine must not approach 1 |
| Teacher-cloud overlap | `tdm_metrics.cloud_overlap` | Recorded only; blur inflates overlap, so it is not a gate |
| Paired MS-SSIM / latent nearest-neighbour | `tdm_metrics.paired_ms_ssim`, `nearest_neighbour_relative_mse` | Forensic only; see the caveat above |

Run the video acceptance report over a completed run:

```bash
PYTHONPATH=. python tests/local_tests/tdm/tools/tdm_video_report.py \
    --run-dir <run root> --out /tmp/video_report.json [--paired-ms-ssim]
```

It reproduces the standalone numbers on the same videos: the warmup-only
step-400 student reads std `88.27` (identical to the standalone
`frame_stats.json`), and the sharpness axis separates the arms the way the
frames do - warmup-only 22-64, teacher 90.5, TDM ladder 192-401 - which is the
exact inverse of the paired-metric ranking (warmup-only `nn_rel_mse 0.606` "in
bounds" versus TDM `1.350` "out").

The step-count preflight sampler and a checkpoint rescorer are deferred to the
first modular Wan validation run, because both need real checkpoints and the
modular LoRA export path resolved.

## Test Scope

```bash
pytest tests/local_tests/tdm/ -v -s
```

| Area | Test | Concern |
|---|---|---|
| Config smoke | `test_tdm_config_smoke.py` | Example YAML parses, resolves `TDMMethod`, and declares expected roles/LoRA knobs without loading weights |
| Flow bridge | `test_tdm_scheduler_math.py` | Mixed-noise transition reconstructs Wan flow noising; invalid direction raises |
| Method wiring | `test_tdm_method_unit.py` | Fake models exercise loss keys, faithful interval support, fake-score-before-generator optimizer ordering, and student/critic updates |
| Upstream parity | `test_tdm_upstream_parity.py` | Assembled context identities plus critic/generator losses and gradients against the upstream-verified transcription |
| Warmup and ladder | `test_tdm_warmup_and_ladder.py` | Wan-family warmup default, student-only gating, CFG teacher regression target, ladder stage resolution and validation, warmup-only versus TDM phase controls |
| H3 joint math | `test_tdm_h3_joint.py` | H3 sigma shifts and grid endpoints, `1 - sigma` model time, and the data-ward velocity sign bridge |
| H3 config | `test_tdm_h3_config_smoke.py` | H3 TDM YAML resolves roles/LoRA, guidance 1.0 with no `cfg_uncond`, ladder and LRs |
| Metrics | `test_tdm_metrics.py` | Sharpness ordering, mode-tightening detection, overlap behaviour, and the documented paired-metric inversion |

## GPU Validation

Runtime validation runs on Slurm (one node, four GB200 GPUs). Launch jobs from
the Slinky login node through
`tests/local_tests/tdm/slurm/launch_h3_tdm_vsa.sh`; `sbatch` is unusable on
this site, so the launcher uses a detached `srun`. The older Kubernetes
runners under `tests/local_tests/tdm/k8s/` are kept for the Wan recipes. Run
outputs belong under an account-writable Lustre directory, not the shared
`/workspace/run` tree.

Suggested local-test command:

```bash
pytest tests/local_tests/tdm/ -v -s
```

Suggested checkpoint smoke command:

```bash
python fastvideo/train/entrypoint/train.py \
    --config examples/train/configs/distribution_matching/wan/tdm_t2v_lora.yaml \
    training.loop.max_train_steps=2 \
    callbacks.validation.every_steps=0
```

For the checkpoint smoke, request the GPU count expected by the config or
override the distributed settings for a smaller smoke. Record run roots,
command output, loss keys, and blockers in the active handoff rather
than in this durable README.
