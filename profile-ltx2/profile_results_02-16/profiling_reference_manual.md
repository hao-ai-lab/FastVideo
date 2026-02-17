# VSA Profiling Onboarding Guide

This is a standalone quick-start manual for profiling VSA vs vanilla FlashAttention in LTX2, with emphasis on two outcomes:
- theoretical speedup (kernel/attention-level)
- actual speedup (end-to-end denoising stage)

Primary example report:
- `profile_results_02-16/denoising_stage_latency.md`

## Goal

When backend, sequence length, frame count, or VSA sparsity changes, produce a comparable report that answers:
1. How much faster is VSA in attention itself?
2. How much faster is the full denoising stage in practice?
3. Where does any speedup gap come from?

## Metrics You Must Produce

1. Theoretical speedup (kernel-level):
- Compare attention kernel latency from trace files.
- Formula: `speedup = FA_kernel_avg / VSA_kernel_avg`

2. Theoretical speedup (attention section-level, optional but recommended):
- Compare a broader section around attention (for example Memcpy-end -> gather-end).
- Formula: `speedup = FA_kernel_avg / VSA_section_avg` (or FA section vs VSA section if available)

3. Actual speedup (pipeline-level):
- Compare denoising stage steady-state average.
- Formula: `speedup = FA_denoise_avg_excl_warmup / VSA_denoise_avg_excl_warmup`

## Data Sources

You need both log and trace artifacts.

1. Stage logs:
- FA log: `profile-ltx2/profile.4.compile.log`
- VSA log: `profile-ltx2/profile.4.vsa.compile.log`

2. Trace files (`*.pt.trace.json.gz`):
- FA traces: `profile-ltx2/traces_20260217_061722/...`
- VSA traces: `profile-ltx2/vsa-traces-0.7/...`

3. Existing scripts:
- `profile_results_02-16/extract_denoising_latency_to_md.py`
- `profile_results_02-16/extract_attention_latency_avg.py`
- `profile_results_02-16/extract_kernel_latency_to_md.py`

## Repository Changes Needed (Checklist)

This section documents what must exist in code for this profiling flow to work.
It is based on current profiling-related diffs in:
- `fastvideo/models/dits/ltx2.py`
- `fastvideo/profiler.py`
- `profile-ltx2/trace_ltx_dit.py`

### 1) DiT forward region must be profile-able

Required:
- A registered profiler region for DiT forward in `fastvideo/profiler.py`:
  - `profiler_region_dit_forward`
- The transformer block loop in `LTXModel._process_transformer_blocks` wrapped by:
  - `with profiler_controller.region("profiler_region_dit_forward")`

Why:
- Without this region, `FASTVIDEO_TORCH_PROFILE_REGIONS=profiler_region_dit_forward`
  will not capture only the denoising-attention area cleanly.

### 2) Attention timing logs must be explicit and consistent

Required in `fastvideo/models/dits/ltx2.py`:
- Log timing for both branches:
  - `>>>> VSA time: ...`
  - `>>>> Vanilla time: ...`
- Timing must synchronize GPU before reading host timer (or use CUDA events).

Important fix to keep:
- Unit mismatch must be resolved/documented.
  - Current logs were printed with `ms` label while values behaved like seconds.
  - Either:
    - print seconds with `s` label, or
    - convert to milliseconds before printing `ms`.
  - Keep extractor logic aligned with whichever format is used.

### 3) Trace driver must set profiling envs predictably

Required in `profile-ltx2/trace_ltx_dit.py`:
- Ability to run both backends (VSA and FA).
- Toggle tracing on/off (`--trace`) and compile mode (`--compile`).
- Set profiler envs when tracing:
  - `FASTVIDEO_TORCH_PROFILER_DIR`
  - `FASTVIDEO_TORCH_PROFILE_REGIONS=profiler_region_dit_forward`
  - wait/warmup/active steps
- Rotate output trace directory when existing one is non-empty.

Why:
- Prevents accidental overwrite and keeps one run = one trace folder.

### 4) Keep profiling code noise low

Recommended:
- Remove temporary debug logs not needed for metrics (for example repeated
  `run_vx/run_ax` prints) to keep log parsing stable.
- If attention timing logs are expensive/noisy, gate them with an env flag.

### 5) Keep parser expectations in sync with logger format

Required for `extract_attention_latency_avg.py`:
- Regex and unit handling must match actual logger output.
- If logger format changes, update parser in the same commit.

Rule:
- Never change log format without updating extraction scripts and this manual.

## Important Unit Note

In your current setup, some log lines print `ms` but the values behave as seconds. Keep this explicit in every report.

## Standard Workflow (Do This Every Run)

### Step 1: Generate Denoising Stage Table

```bash
python3 profile_results_02-16/extract_denoising_latency_to_md.py \
  /home/hal-jundas/codes/FastVideo-demo/profile-ltx2/profile.4.compile.log \
  /home/hal-jundas/codes/FastVideo-demo/profile-ltx2/profile.4.vsa.compile.log \
  -o /home/hal-jundas/codes/FastVideo-demo/profile-ltx2/profile_results_02-16/denoising_stage_latency.md
```

Use `avg_excl_warmup` for actual speedup.

### Step 2: Extract Attention E2E Averages From Logs

```bash
python3 /home/hal-jundas/codes/FastVideo-demo/profile-ltx2/profile_results_02-16/extract_attention_latency_avg.py \
  --vsa-min-s 0.005 --vsa-max-s 0.05 \
  --fa-min-s 0.005 --fa-max-s 0.05
```

This gives steady-state attention timing from app logs.

### Step 3: Extract Kernel-Level Trace Stats

```bash
python3 /home/hal-jundas/codes/FastVideo-demo/profile-ltx2/profile_results_02-16/extract_kernel_latency_to_md.py
```

Output:
- `profile_results_02-16/kernel_latency_summary.md`

### Step 4: Optional Section-Level Trace Timing

If kernel-only speedup and actual speedup diverge, compute section timings around attention boundaries (for example Memcpy-end -> gather-end) to capture overhead outside the focal kernel.

## Example (From `denoising_stage_latency.md`)

From the current report:

- Denoising stage averages (excl warmup):
  - FA: `7200.80759 ms`
  - VSA: `5754.60869 ms`
  - actual speedup: `7200.80759 / 5754.60869 = 1.2513x`

- Attention E2E averages (steady-state):
  - FA: `12.999318 ms`
  - VSA: `9.534198 ms`
  - attention E2E speedup: `1.3634x`

- Trace kernel averages:
  - FA flash kernel: `9.534580 ms`
  - VSA `_attn_fwd_sparse`: `2.762085 ms`
  - kernel theoretical speedup: `3.4520x`

- Trace section example (VSA Memcpy-end -> gather-end):
  - VSA section avg: `5.513097 ms`
  - vs FA kernel `9.534580 ms` gives section-level speedup indicator: `1.7294x`

Interpretation pattern:
- Kernel speedup is highest.
- Attention E2E speedup is smaller.
- Denoising-stage speedup is smaller still.
- This usually means non-kernel overhead (data movement, prep/postprocess, collectives, synchronization) absorbs part of the theoretical gain.

## Required Run Metadata (Always Record)

Add this block to each new report:

- date/time
- git commit hash
- model checkpoint
- backend (`FLASH_ATTN` or `VIDEO_SPARSE_ATTN`)
- VSA sparsity settings
- sequence length / effective token count
- frame count, resolution, batch size
- denoising steps
- GPU type/count and SP/TP/world size
- trace file paths and log file paths

## Recommended Report Structure

For each experiment, publish one markdown file with:

1. Experiment settings table
2. Denoising stage table (`run_01...run_N`, `avg_all_rounds`, `avg_excl_warmup`)
3. Attention E2E summary (FA vs VSA)
4. Kernel summary table (occurrence/min/max/avg)
5. Section timing table (with explicit start/end boundary definitions)
6. Speedup summary table:
- theoretical kernel speedup
- attention E2E speedup
- actual denoising-stage speedup
7. Notes (unit caveats, anomalies, warmup behavior)

## Quick Checklist For New Interns

Before profiling:
- confirm backend and sparsity configuration
- decide fixed workload shape for fair comparison
- ensure logging and trace capture are enabled

After profiling:
- run all extract scripts
- verify units
- compute three speedups (kernel / attention E2E / denoising)
- document boundaries used for section timings
- store logs/traces + markdown report together
