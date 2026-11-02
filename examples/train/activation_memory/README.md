# Wan activation-memory profiling on Ascend

These examples measure public Wan2.1-1.3B through FastVideo's WanModel/FSDP2
adapter. They collect block costs and full-training memory lifetimes, export a
profile, and execute the resulting activation plan. The measured workload uses
FP32 master parameters, BF16 attention, batch size one per rank, and real video
latents with public UMT5 text conditions. Single-role updates minimize
flow-matching MSE at timestep 500 using a fixed noisy input and target; DMD2
uses the production method's seeded noise and timestep sampling.

Use a local copy of `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, a working CANN/torch-npu
environment, and an idle NPU. The sample encoder additionally uses Diffusers,
Transformers, OpenCV, and PyArrow. SciPy provides the planning solver.

## Prepare a video sample

Set `WAN_MODEL` to the local model directory and `VIDEO` to a video file. The
public robot-pouring video is available at
[FastVideo's pinned example](https://raw.githubusercontent.com/hao-ai-lab/FastVideo/b0047cfedc081b31ccc9efc801a5edb8474e15d2/assets/videos/robot_pouring.mp4).

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
mkdir -p measurements
python examples/train/activation_memory/prepare_sample.py \
  --model "$WAN_MODEL" --video "$VIDEO" --frames 17 --height 256 --width 448 \
  --prompt 'A robot arm pours liquid into a cup on a table.' \
  --output measurements/robot.pt
python examples/train/activation_memory/prepare_sample.py \
  --model "$WAN_MODEL" --video "$VIDEO" --text \
  --prompt 'A robot arm pours liquid into a cup on a table.' \
  --output measurements/robot.pt
```

The sample stores normalized latents `[B,16,T,H,W]`, padded text embeddings,
negative-prompt embeddings, source frame indices, and spatial dimensions.
Encoding and model initialization are outside training-step timing.

## Profile and execute a single role

```bash
python examples/train/activation_memory/profile_wan.py profile \
  --model "$WAN_MODEL" --training-sample measurements/robot.pt \
  --warmup 5 --steps 5 --output measurements/wan_recompute_profile.json
python examples/train/activation_memory/profile_wan.py profile \
  --model "$WAN_MODEL" --training-sample measurements/robot.pt \
  --reference-profile measurements/wan_recompute_profile.json --policy save \
  --warmup 5 --steps 5 --output measurements/wan_profile.json
python examples/train/activation_memory/profile_wan.py benchmark \
  --model "$WAN_MODEL" --training-sample measurements/robot.pt \
  --profile measurements/wan_profile.json --budget-gib 24 \
  --warmup 5 --steps 5 --output measurements/wan_budget.json
```

Profiling records continuous intervals between block pre-hooks, including FSDP
parameter transitions. The full-save calibration supplies a second reference;
the fixed memory vector is the per-event envelope of the residual storage from
both references. The block choices retain their separately measured activation
and unsharded-parameter lifetimes. Both references use the calibration input.

The benchmark writes the policy to `wan_budget.plan.json` and records every
complete training step, peak allocated bytes, peak reserved bytes, and planning
time. Baselines use `--policy save`, `recompute`, `ops`, `offload`, or
`block_skip`. Prefix, suffix, and uniform layouts also accept `--save-count`.
Use a separate input for manual-layout tuning and final evaluation.

Two-rank profiling and execution use `torchrun --standalone --nproc-per-node=2`
with the same entrypoint. Set `ASCEND_RT_VISIBLE_DEVICES` to the two selected
devices. Profiles combine the maximum rank-local storage at each event; the
budget applies to each rank. The benchmark reports the maximum step time and
memory across ranks and retains the individual rank measurements. Reprofile
when changing the rank count or video shape.

The exported profile can also be used by the regular training configuration:

```yaml
training:
  model:
    activation_memory_profile: measurements/wan_profile.json
    activation_memory_budget_gib: 24
    activation_memory_plan_output: measurements/wan_plan.json
```

## Profile the DMD2 role schedule

The DMD2 example executes the production builder, dataloader, data-latent
rollout, student and critic updates, and a frozen teacher. Its AdamW settings
match the method configuration. Convert the same sample to the training
parquet format, then measure the full role schedule using the block profile.

```bash
python examples/train/activation_memory/prepare_parquet.py \
  --model "$WAN_MODEL" --sample measurements/robot.pt \
  --output measurements/robot_parquet/train.parquet
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
python examples/train/activation_memory/profile_dmd2.py \
  --model "$WAN_MODEL" --training-sample measurements/robot.pt \
  --data measurements/robot_parquet --calibrate-from measurements/wan_profile.json \
  --warmup 5 --steps 5 --output measurements/dmd_profile.json
python examples/train/activation_memory/profile_dmd2.py \
  --model "$WAN_MODEL" --training-sample measurements/robot.pt \
  --data measurements/robot_parquet --profile measurements/dmd_profile.json \
  --budget-gib 54 --warmup 5 --steps 5 --output measurements/dmd_budget.json
```

The role profile includes optimizer temporaries and overlapping student/critic
graphs. Keep the allocator setting fixed across its calibration and baselines.
Profile acquisition, manual tuning, and solver time are separate costs from
steady-state training steps.

### Measure complete-step role costs

Use an independent calibration sample to measure the full-recompute reference
and four configurations that change only one role to save or offload. For
example, `--student-policy save` leaves the critic at the default recompute
policy. `--critic-policy offload` changes only the critic.

Fit these complete-step increments into the existing memory profile:

```bash
python examples/train/activation_memory/fit_role_costs.py \
  --profile measurements/dmd_profile.json \
  --reference measurements/recompute.json \
  --student-save measurements/student_save.json \
  --student-offload measurements/student_offload.json \
  --critic-save measurements/critic_save.json \
  --critic-offload measurements/critic_offload.json \
  --output measurements/dmd_role_costs.json
```

For each role, the fitter divides its whole-step change by its block count and
subtracts that role's minimum increment, leaving non-negative variable costs.
Memory and host-storage vectors remain unchanged. The profile metadata records
the measured increments and the common `fixed_step_ms` term. The plan's
`time_ms` is the variable MILP objective; adding `fixed_step_ms` gives the
additive complete-step estimate. Freeze this profile before evaluating new
inputs. This approximation still omits interactions between role policies.

## Export and sample updated weights

Add `--save-model measurements/trained` to the single-role benchmark to export
its updated transformer through FastVideo's distributed checkpoint exporter.
Export happens after the timed training steps. With five warmup and five
measured updates, the inference checkpoint is `inference/checkpoint-10`.
The export metadata records the training sample, policies, budget, rank count,
seed, and optimizer settings.

```bash
python examples/train/activation_memory/sample_checkpoint.py \
  --model "$WAN_MODEL" \
  --trained-transformer measurements/trained/inference/checkpoint-10/transformer \
  --training-sample measurements/robot.pt --steps 30 --seed 17 \
  --output measurements/trained_video.mp4
```

Sampling loads the exported transformer with the public VAE and saved text
embeddings. Its JSON records the prompt, checkpoint, scheduler, sampling seed,
and output shape.

## Limitations

Budgets constrain allocated bytes. Reserved allocator storage and driver
allocations also consume physical memory. The examples have been exercised on
Ascend 910B2C; their profiles are specific to the model, shape, precision,
optimizer, attention backend, and parallel layout. A short training run checks
memory and execution behavior, not convergence or generation quality.
