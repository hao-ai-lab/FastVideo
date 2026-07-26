# LTX-2 fixed-arena CUDA-graph gate (4x GB200)

## Verdict

The complete two-step fixed-arena training graph remains **unmeasured and blocked**. Both capture attempts failed before replay with the same pageable CPU-to-CUDA copy error. There is no graph latency, saving, MFU, replay correctness, RNG-variation, or graph-pool peak-memory result, so this is **not a 50% MFU claim** and the required >=32 ms saving gate cannot be evaluated.

No FastVideo repository files or PR branches were changed. The original fixed-arena probe was preserved.

## Exact environment

- dlcluster job: `1619868`
- topology: 4x GB200 (SM100)
- FastVideo worktree: `/mnt/fv-pr1630-7f13`
- FastVideo SHA: `7f139e2b28610063d2f30526ba8f0ccae5d88944`
- Python environment: `/mnt/FastVideo/.venv`
- source override: `PYTHONPATH=/mnt/fv-pr1630-7f13`
- config: `examples/train/configs/overfit_ltx2_t2v.yaml`
- data: `/mnt/FastVideo/data/ltx2_overfit_preprocessed`
- graph shape: two consecutive complete training steps per replay; only the second/final parameter all-gather is joined
- planned schedule: 5 eager warmups, 5 same-process eager controls, 5 warm replays, 20 measured replays
- comparison baseline: `403.724612435326 ms/step` (`35.7771% MFU`) from the exact fixed-arena probe
- acceptance gate: at least `32 ms/step` saved

## Attempt 1: original static batch

Command:

```bash
FV_JOBID=1619868 .agents/skills/run-fastvideo-dlcluster/dlrun --profile gb200 'cd /mnt/fv-pr1630-7f13 && unset TORCH_LOGS && PYTHONPATH=/mnt/fv-pr1630-7f13 OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=4 /mnt/zero2_ltx2_graph_probe.py --training.data.data_path /mnt/FastVideo/data/ltx2_overfit_preprocessed 2>&1 | tee /mnt/zero2_ltx2_graph_probe_2step_exact_data.log'
```

All four ranks failed during the first `_run_step` inside `torch.cuda.graph(...)`:

```text
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned. Please use tensor.pin_memory() or allocate the tensor with pin_memory=True.
```

Per-rank memory at failure:

- allocated: `97.23348999023438 GiB`
- reserved: `97.666015625 GiB`
- peak allocated: `97.23451280593872 GiB`

The probe catches capture exceptions at the graph boundary and emits `GRAPH_BLOCKER`, so the log contains no deeper Python stack or exact offending operator.

## Attempt 2: one bounded pinned-batch retry

Only the static input batch changed: every CPU Tensor reachable through its dict/list/tuple containers was replaced by pinned storage, and that one pinned object remained alive through eager control, capture, and the intended replays. No other design changed.

Command:

```bash
FV_JOBID=1619868 .agents/skills/run-fastvideo-dlcluster/dlrun --profile gb200 'cd /mnt/fv-pr1630-7f13 && unset TORCH_LOGS && PYTHONPATH=/mnt/fv-pr1630-7f13 OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=4 /mnt/zero2_ltx2_graph_probe_pinned.py --training.data.data_path /mnt/FastVideo/data/ltx2_overfit_preprocessed 2>&1 | tee /mnt/zero2_ltx2_graph_probe_2step_pinned.log'
```

The exact model and data loaded, the fixed arena initialized, and eager warmup/control completed by control flow. The script does not print eager records before successful graph completion, so **no eager-control timing is present in the log**. Capture then failed identically on all four ranks, with the same memory values and exception shown above. This means another pageable CPU tensor is created or retained somewhere inside the training-step path (or inside a custom input object not traversed by the narrow pin helper). Its exact origin was not localized because the authorized retry budget ended here.

## Correctness and memory status

- capture completed: no
- replay started: no
- graph latency/MFU: unavailable
- eager-control timing: unavailable in logs
- loss/gradient finiteness after replay: not run
- graph-safe RNG variation: not run
- optimizer-step/parameter-change checks: not run
- replica/master consistency checks after replay: not run
- graph-pool peak: unavailable
- OOM: no; failure occurred at approximately `97.23 GiB` allocated/rank

## SHA-256 provenance

```text
7ff05aafe045a53e754a88c4842fe7be33606440e2b1a8f18718f2793c24fff6  /mnt/zero2_ltx2_probe.py
4cda0ba6ad0f55912be3eaf2fabd342ac84d96b115ddc51ccf5eb5636795938c  /mnt/zero2_ltx2_graph_probe.py
e5d693aa5464907ad9a33a38889a55802ee1ceab707bfcc78dc11544386c2216  /mnt/zero2_ltx2_graph_probe_pinned.py
9f39f85c07344a94b8a73ecad0bf844764fe560fb752966b8feaef50c4035aba  /mnt/zero2_ltx2_graph_probe_2step_exact_data.log
b9787d89b5ac6cd8fc118fc0abb67e78c3ce14c0730d4d4152e23af54e5a0706  /mnt/zero2_ltx2_graph_probe_2step_pinned.log
```
