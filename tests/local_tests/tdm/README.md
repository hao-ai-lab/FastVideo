# TDM Local Tests

Local-only tests for the Trajectory Distribution Matching implementation under
`fastvideo.train.methods.distribution_matching.tdm`.

This is a Wan flow-matching adaptation of the original CogVideoX/diffusion TDM
reference, not a checkpoint-compatible port of the reference training script.
The tests in this directory focus on the math bridge and method wiring. Full
Wan training validation must run on Modal L40S.

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

## Test Scope

```bash
pytest tests/local_tests/tdm/ -v -s
```

| Area | Test | Concern |
|---|---|---|
| Config smoke | `test_tdm_config_smoke.py` | Example YAML parses, resolves `TDMMethod`, and declares expected roles/LoRA knobs without loading weights |
| Flow bridge | `test_tdm_scheduler_math.py` | Mixed-noise transition reconstructs Wan flow noising; invalid direction raises |
| Method wiring | `test_tdm_method_unit.py` | Fake models exercise loss keys, timestep support, fake-score weights, and student/critic backward routing |

## Modal Validation

Run from branch `interleavethinker` using
`fastvideo/tests/modal/launch_l40s_job.py`, applying this branch as the patch.

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
override the distributed settings for a smaller smoke. Record Modal app IDs,
command output, loss keys, and blockers in the issue handoff or PR body rather
than in this durable README.
