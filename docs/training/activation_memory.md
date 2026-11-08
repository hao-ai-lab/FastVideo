# Activation memory budgets

The Wan training adapter can choose a different activation policy for each
transformer block from a measured memory profile. The planner minimizes block
execution time subject to the device budget at every recorded execution event.
It supports saving activations, full recomputation, selective checkpointing,
and CPU offload. Model parameters, gradients, optimizer state, and allocations
outside the planned blocks are included as fixed storage in the profile.

Set the budget and profile under `training.model`:

```yaml
training:
  model:
    activation_memory_budget_gib: 48.0
    activation_memory_profile: measurements/wan_profile.json
    activation_memory_plan_output: measurements/wan_plan.json
```

`activation_memory_host_budget_gib` optionally bounds activation storage on the
CPU. The budget uses GiB (2^30 bytes). Omit the manual
`enable_gradient_checkpointing_type` when enabling the planner. Roles with
planned blocks must use an adapter that explicitly declares
`activation_memory_plan` in its constructor. Unsupported adapters are rejected
before the plan is exported or any model is loaded. The Wan adapter applies
the plan before FSDP wrapping.

## Profiles and role lifetimes

A `MemoryProfile` contains ordered execution events, fixed device storage at
each event, and policy alternatives for each `(role, block)` pair. Each
alternative supplies measured time and device/host storage vectors over the
same events. Include temporary recomputation and transfer allocations at the
events that execute them. Retained activation storage extends from its forward
producer through its backward consumer.

For FSDP mixed precision, include unsharded parameter copies retained by the
backward graph in the policy-dependent storage. Saving a block can keep its
BF16 weights alive after forward, whereas recomputation can release and
recreate them. Master parameters remain in fixed storage.

For DMD2, the profile should follow the method's actual order: student rollout,
critic and teacher scoring, critic training forward, and both backwards.
Activations that overlap in this order share an event constraint; disjoint
lifetimes can reuse the same device budget. Frozen teacher parameters and its
temporary scoring allocations belong in fixed storage.

The solver selects exactly one alternative for each block. A plan records the
chosen policies, predicted per-event device and host storage, measured block
time, device and optional host budgets, and profile metadata. Infeasible budgets raise an error before
models load. Rank zero writes the exported plan.

Use `MemoryProfile.save` and `MemoryProfile.load` for JSON profiles, and
`plan_activation_memory` to inspect a plan without constructing models. Include
the model revision, input shape, precision, attention backend, optimizer,
parallel layout, and calibration data in profile metadata. Calibrate and
evaluate on separate inputs with the same execution shapes.

The [Ascend Wan profiling examples](../../examples/train/activation_memory/README.md)
encode public video/text inputs, measure native WanModel/FSDP2 block costs and
full-step memory events, and execute a budget plan. They include one- and
two-rank profiling and the production DMD2 student/critic/teacher schedule.
The DMD2 cost-fitting example measures complete-step increments by changing one
role's policy at a time. It separates a common time term from non-negative
variable policy costs; exported metadata states how to recover the estimated
complete-step time from the plan's objective.

## Training state

The execution layer uses PyTorch's checkpoint wrapper and activation-wrapper
state hooks. Plans preserve model state-dict keys and parameter order, so
optimizer state can resume with a different activation policy. Recomputation
preserves RNG state. Offload selects pinned memory for the block's device type.

## Measured Wan training on Ascend

Native WanModel/FSDP2 measurements use public Wan2.1-1.3B weights, FP32 master
parameters, BF16 execution, one video per rank, and AdamW with `foreach=False`.
Robot video supplies memory calibration, Cosmos video selects manual layouts,
and a hiker video supplies final evaluation. Each row contains three independent
processes with five warmup and five measured updates per process. Complete-step
means include backward execution, optimizer updates, and synchronization.

| Frames | Device budget, GiB | Policy | Mean step, ms | Peak allocated, GiB | Peak reserved, GiB |
|---|---:|---|---:|---:|---:|
| 17 | 24 | Full recomputation | 554.91 | 21.640 | 24.229 |
| 17 | 24 | Tuned manual | 463.15 | 23.777 | 32.186 |
| 17 | 24 | Automatic | 471.82 | 23.498 | 31.727 |
| 33 | 28 | Full recomputation | 661.24 | 21.872 | 26.422 |
| 33 | 28 | Tuned manual | 575.77 | 27.988 | 38.123 |
| 33 | 28 | Automatic | 579.67 | 27.479 | 37.516 |
| 49 | 28 | Full recomputation | 904.77 | 22.148 | 27.797 |
| 49 | 28 | Tuned manual | 819.87 | 27.736 | 38.943 |
| 49 | 28 | Automatic | 829.76 | 26.129 | 38.014 |

The automatic plans save 25, 22, and 13 of the 30 blocks respectively. Their
means are 14.97%, 12.34%, and 8.29% shorter than full recomputation. Tuned manual
layouts have slightly shorter means and use more allocated memory. The tested
video grid is 17, 33, and 49 frames at 256×448; at a common 28 GiB threshold,
full-save execution reaches 17 frames and the other three strategies reach 49.
This is a longest-measured-length comparison rather than an exact capacity limit.

Two-rank FSDP2 with one video per rank satisfies a 14 GiB per-rank budget:
the automatic plan saves 18 blocks, uses 13.524 GiB, and averages 516.89 ms.
The independently tuned manual plan uses 13.799 GiB and averages 516.87 ms;
full recomputation averages 574.45 ms. Times and peaks take the maximum across
ranks. The global batch is two, so this is a separate budget experiment rather
than a fixed-global-batch scaling comparison.

Production DMD2 uses a frozen teacher, trainable student and critic, and a
54 GiB allocated budget. The complete-step role-cost plan saves 24 student
blocks and all 30 critic blocks, uses 53.142 GiB, and averages 2757.26 ms.
The independently tuned periodic layout uses the same allocated peak and
averages 1318.19 ms. An isolated-block-cost ablation averages 2816.09 ms.
Actual single-role increments calibrated on Cosmos predict 1192.46 ms for the
joint plan; the measured complete-step cost is substantially higher. All DMD2
configurations use the same optimizer and `expandable_segments:True` setting.

CPU tests check optimal selection against enumeration, role lifetimes,
host constraints, configuration, gradients, AdamW updates, dropout RNG, and
checkpoint restoration, adapter support, exported constraints, and cross-rank
measurement aggregation. Native FSDP restores all 3,300 model and optimizer
state tensors exactly. A two-rank automatic-plan checkpoint after ten updates
reloads and generates a finite 17-frame video with public UMT5 conditions and
VAE. Export and generation are outside the training timing interval.

[Raw measurements](../../examples/train/activation_memory/measurements_910b2c.json)
include every final step, rank-local measurements, frozen profiles, selected
plans, and independent manual-tuning decisions. The profiling examples reproduce
the workload and separate profile acquisition, planning, and steady-state timing.
Multi-rank DMD2 reports retain every rank's records and take the maximum step
time, allocated memory, and reserved memory across ranks for each step.

## Limitations

The budget constrains profiled allocated bytes. Reserved allocator storage and
driver allocations also consume physical memory. Workspace or shape changes
require a matching profile. The solver's objective sums measured block times; transfer
overlap can change actual step time. In the measured DMD2 workload, both
isolated-block costs and calibrated single-role increments mis-rank the joint
execution against the manual baseline. Independently constructed native BF16
models also show gradient/update variation, including the all-save control;
exact state restoration does not imply bit-exact next-step execution. Validate the exported plan's peak memory
and step time on the complete workload before choosing a production budget.

The configuration integration currently targets Wan. The profile format can
express multiple roles, but each training schedule and parallel layout needs
its own measured profile.
