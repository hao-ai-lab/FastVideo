# Select a DMD2 layout under an allocated-memory budget

Use the single-rank DMD2 profile for the same model, training shape, optimizer,
backend and allocator as the benchmark. The selector evaluates 56 pairs of
student and critic save counts for 30-block Wan. For each pair it adds the
selected block costs to fixed storage at every event. When more than 40 pairs
fit, it benchmarks the feasible subset of a smaller 4 × 5 grid.

```bash
export PYTHONPATH="$PWD"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
python examples/train/activation_memory/select_joint_plan.py \
  --model "$WAN_MODEL" --profile measurements/dmd_profile.json \
  --training-sample measurements/cosmos.pt --data measurements/cosmos_parquet \
  --budget-gib 54 --output measurements/grid/matrix.json
```

Each candidate runs in a fresh process with two warmups and five timed updates.
For a host with a device availability check, pass its shell script with
`--launcher`. Use a separate output directory per tuning run. The matrix
records predicted feasibility for every pair and observed timing and memory
for each successful run. When a benchmark exits with a nonzero code, the matrix
records its failure, exit code and log path, and tuning continues. The winner has the lowest mean complete-step time
among successful runs whose observed peak also satisfies the budget. If none
qualify, the matrix retains all results with a null winner and the selector exits
with status 1.

Freeze the winning counts and evaluate on a separate sample and its matching
parquet, using the same model and allocator settings.

```bash
python examples/train/activation_memory/profile_dmd2.py \
  --model "$WAN_MODEL" --training-sample measurements/hiker.pt \
  --data measurements/hiker_parquet --policy uniform --save-count "$STUDENT" \
  --critic-save-count "$CRITIC" --warmup 5 --steps 5 --output measurements/heldout.json
```

In the 17-frame, 54 GiB Ascend experiment, 54 pairs passed the memory model
and 18 were measured on Cosmos. The winning 24+16 layout averaged 1300.42 ms
in three independent hiker processes at 53.142 GiB peak allocated memory.
The additive solver averaged 2757.26 ms and the tuned manual layout averaged
1318.19 ms on that evaluation workload.
