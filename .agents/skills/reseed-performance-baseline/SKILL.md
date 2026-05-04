---
name: reseed-performance-baseline
description: Re-seed the HF performance-tracking baseline for an intentional runtime, dependency, or environment-caused benchmark shift. Use when performance CI fails because metrics such as latency, throughput, component time, or peak memory changed for an accepted reason and the rolling median baseline in FastVideo/performance-tracking must be advanced by replicating one reviewed shifted source result into three success=true records, or five records when explicitly requested.
---

# Re-seed Performance Baseline

## Purpose

Replace or advance the rolling performance baseline for a single
`(model_id, gpu_type)` pair in the HF dataset
`FastVideo/performance-tracking`.

Performance comparison uses the median of up to the last 5 successful records
for the same model and GPU. Failed records are useful audit history, but they
do not move the future baseline because `compare_baseline.py` loads records
with `successful_only=True`.

For a 5-record median, one shifted record is not enough to move the median if
the other four records are from the old runtime. This skill therefore creates
3 reviewed `success=true` records from one accepted shifted source result by
default. If the user explicitly asks for a full reset, create 5 records.

These replicated records are an intentional operator-approved baseline reset,
not independent measurements. Mark them clearly with provenance fields so the
HF history remains auditable.

Use this skill when a performance test fails for an intentional and reviewed
reason, such as a torch/runtime/container upgrade that legitimately increases
peak memory or changes timings. This is the performance equivalent of
`reseed-ssim-references`: backup first, scope tightly, require explicit human
approval, then upload reviewed accepted baseline records.

## When to use

- A PR or main run failed the rolling performance comparison by more than the
  allowed regression threshold, and maintainers agree the shift is caused by
  an intentional runtime, dependency, hardware image, or benchmark environment
  change rather than a FastVideo logic regression.
- One shifted source result has been reviewed and accepted, and the operator
  wants to replicate it into 3 successful records so the rolling median moves
  immediately. Use 5 records only when the user explicitly asks to fully reset
  the last-5 window.

## When not to use

- The benchmark failure might be a real code regression. Fix or investigate
  the code path first.
- The fixed benchmark thresholds in
  `.buildkite/performance-benchmarks/tests/*.json` are too low. Those are a
  separate gate from the rolling HF baseline and may need a code review change.
- There is no clear source run, commit, and rationale. Baseline history is a
  production signal; do not edit it without provenance.

## Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `model_id` | Yes | Benchmark id, e.g. `wan-t2v-1.3b-2gpu`. This maps to the HF subdirectory after `sanitize(model_id)`. |
| `gpu_type` | Yes | Exact GPU device string from the performance record, e.g. the L40S device name emitted by CI. Baselines are GPU-specific. |
| `source_result` | Yes | Path or Buildkite artifact URL for one accepted shifted performance JSON. Buildkite uploads raw `perf_*.json` artifacts for failed performance jobs when result files exist. |
| `replica_count` | No | Number of success records to create from `source_result`. Default: `3`. Only use `5` if the user explicitly asks for a full reset. |
| `intent_rationale` | Yes | One-line explanation for why the baseline shift is legitimate. This is written into provenance and should be reused in the PR. |

Hardcoded defaults:

- HF repo: `FastVideo/performance-tracking` (`HF_REPO_ID` override is
  supported by the code, but use the default unless the user explicitly asks).
- Local sync root: `/tmp/perf-tracking` or a timestamped local backup under
  `performance_reseed_backup/`.
- Baseline window: last 5 `success=true` records for the same
  `(model_id, gpu_type)`.
- Default reseed count: 3 replicated `success=true` records from one reviewed
  source result. Explicit full-reset count: 5.

## Steps

### 1. Validate the target and source result

If `source_result` is a Buildkite artifact URL, download it first into a
local scratch directory such as `performance_reseed_source/` and use that
downloaded JSON path for the rest of the workflow. If the agent cannot access
the artifact because Buildkite authentication is missing, ask the user to
download the artifact manually and provide the local path.

Confirm the source JSON exists and normalize it the same way
`compare_baseline.py` does:

- `model_id` comes from `benchmark_id`.
- `gpu_type` comes from `device`.
- `memory` comes from `max_peak_memory_mb`.
- `latency` comes from `avg_generation_time_s`.
- `throughput` comes from `throughput_fps`.
- component timings are `text_encoder_time_s`, `dit_time_s`, and
  `vae_decode_time_s`.

Stop if the source result's `benchmark_id` or `device` does not match the
requested `model_id` and `gpu_type`.

Set `replica_count` to `3` by default. Set it to `5` only when the user
explicitly asks to upload the same shifted source result 5 times for a full
last-5 reset. Reject other counts unless the user gives a concrete reason.

Check that `HF_API_KEY` is exported. The sync path may be public, but the
upload path requires write access.

### 1a. How to obtain `source_result` from CI

The performance CI exports raw source results for failed performance jobs when
result files exist. The artifact comes from:

```text
fastvideo/tests/performance/results/perf_*.json
```

and is uploaded by Buildkite as a raw performance result artifact. The normal
operator flow is:

1. Open the failed Buildkite performance job.
2. Download the `perf_*.json` artifact for the failed benchmark.
3. Pass the local path or artifact URL as `source_result`.

Do not scrape the Markdown performance summary to reconstruct the JSON. The
raw `perf_*.json` is the source of truth for normalization and provenance.
If no raw JSON artifact is present, the benchmark likely failed before writing
results, so that run is not a valid source for baseline reseeding.

### 2. Sync and back up existing HF records

Use `fastvideo/tests/performance/hf_store.py` helpers directly. Do **not** use
`compare_baseline.py` as a sync shortcut; on full main runs it can persist
records, while this step must only fetch and back up existing history.

The sync command pattern is:

```bash
export PERFORMANCE_TRACKING_ROOT="${PERFORMANCE_TRACKING_ROOT:-/tmp/perf-tracking}"
export HF_REPO_ID="${HF_REPO_ID:-FastVideo/performance-tracking}"
PYTHONPATH=fastvideo/tests/performance python -c 'from hf_store import sync_from_hf; import os; sync_from_hf(os.environ["PERFORMANCE_TRACKING_ROOT"], strict=True)'
```

Then back up only the sanitized model directory:

```bash
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
MODEL_SAFE=$(python - <<'PY'
from fastvideo.tests.performance.hf_store import sanitize
print(sanitize("<model_id>"))
PY
)
BACKUP_DIR="performance_reseed_backup/${TIMESTAMP}_${SHORT_COMMIT}_${MODEL_SAFE}"
mkdir -p "$BACKUP_DIR"
cp -R "${PERFORMANCE_TRACKING_ROOT}/${MODEL_SAFE}" "$BACKUP_DIR/" 2>/dev/null || true
```

Write provenance next to the backup:

```bash
cat > "$BACKUP_DIR/PROVENANCE.txt" <<EOF
model_id: <model_id>
gpu_type: <gpu_type>
source_result: <source_result>
replica_count: <3_or_5>
head_commit: $(git rev-parse HEAD)
timestamp_utc: $(date -u +%FT%TZ)
reason: <intent_rationale>
EOF
```

If the backup has no prior records, this is not a destructive reseed; it is a
first baseline seed. Continue, but report that baseline history was empty.

### 3. Compute old baseline and candidate shift

Load the last 5 successful records for the target:

```python
from fastvideo.tests.performance.hf_store import load_records_for_model

records = load_records_for_model(
    "/tmp/perf-tracking",
    "<model_id>",
    "<gpu_type>",
    last_n=5,
    successful_only=True,
)
```

Print a small table showing the source result metrics, the replicated
candidate median, and the old medians for:

- `latency`
- `throughput`
- `memory`
- `text_encoder_time_s`
- `dit_time_s`
- `vae_decode_time_s`

Also print how many successful old records exist. Make clear:

- 1 shifted record only seeds audit history and usually does not move the
  median.
- 3 replicated shifted records in a 5-record window move the median
  immediately.
- 5 replicated shifted records fully reset the rolling window to the source
  result's runtime profile.
- Replicated records are not independent measurements; they are an intentional
  approved baseline reset and must be labeled that way.

### 4. Confirm intent

Require an explicit confirmation phrase before preparing the upload:

> About to RE-SEED performance baseline for `<model_id>` on `<gpu_type>`.
> This will upload `<N>` new `success=true` records to
> `FastVideo/performance-tracking/<sanitize(model_id)>/`.
>
> Reason: `<intent_rationale>`
> Source result: `<source_result>`
> Replica count: `<replica_count>`
> Note: these records replicate one reviewed measurement to force the rolling
> median to the accepted runtime profile.
> HEAD: `<git rev-parse --short=12 HEAD>`
> Backup: `<BACKUP_DIR>`
>
> Reply `confirm performance reseed` to proceed, anything else to abort.

Do not continue unless the user types exactly `confirm performance reseed`.

### 5. Create the accepted seed records

Create `replica_count` normalized records from the single source result. Each
record must include:

- `model_id`
- `timestamp`
- `commit_sha`
- `gpu_type`
- `latency`
- `throughput`
- `memory`
- `text_encoder_time_s`
- `dit_time_s`
- `vae_decode_time_s`
- `success: true`

Optional provenance fields are allowed and useful:

- `baseline_reseed: true`
- `baseline_reseed_reason`
- `baseline_reseed_source_result`
- `baseline_reseed_source_timestamp`
- `baseline_reseed_replicated_source: true`
- `baseline_reseed_batch_size`
- `baseline_reseed_batch_index`
- `baseline_reseed_operator`

Use a fresh reseed timestamp for each replicated record, not the original
source result timestamp. This is required because
`load_records_for_model(..., last_n=5)` keeps the last records after loading
the model directory; stale filenames/timestamps may not enter the last-5
window and therefore may not move the median. Preserve the original source
timestamp in `baseline_reseed_source_timestamp`.

Use the existing filename convention from `_write_tracking_record()`:
`<sanitize(timestamp)>_<sanitize(commit_sha)>.json` under the sanitized model
directory, but include a deterministic suffix such as `_reseed_01`,
`_reseed_02`, and `_reseed_03` before `.json` so the replicated files do not
overwrite each other. For a 5-record full reset, continue through
`_reseed_05`.

If the source record already exists on HF with `success=false`, do not edit it
in place unless the user explicitly asked for an audit-preserving correction.
Prefer uploading new accepted seed records so failed history remains visible.

### 6. Pause before upload

Print:

- Backup directory path.
- HF paths that will receive the new records.
- Old rolling medians.
- Source metrics, replica count, and candidate median.
- Rationale.

Ask the user to reply exactly `upload`. Anything else aborts and leaves the
prepared records plus backup on disk.

### 7. Upload only the scoped records

Use the shared storage helper so the path and repo type match CI:

```python
from fastvideo.tests.performance.hf_store import upload_record

upload_record("<local_record_path>", record, strict=True)
```

Run it once per prepared record. Each upload goes to:

```text
FastVideo/performance-tracking/<sanitize(model_id)>/<record_filename>.json
```

Never bulk upload the whole tracking root. Never modify another model's
directory in the same operation.

### 8. Report outcome

Report:

- Uploaded HF paths.
- Backup directory.
- Old baseline window count and medians.
- Source metrics, replica count, and candidate median.
- Expected effect: 3 replicated shifted records move the 5-record median; 5
  replicated shifted records fully reset the window to the accepted source
  result.
- Any separate threshold changes still needed in
  `.buildkite/performance-benchmarks/tests/*.json`.

Include the `intent_rationale` in the PR or follow-up comment so reviewers can
distinguish an accepted baseline shift from a hidden regression.

## Failure modes and handling

- **`HF_API_KEY` unset.** Stop before upload. Do not create an untracked
  process that appears to have reseeded but never reached HF.
- **Source result does not match target.** Stop. The wrong benchmark or GPU
  would poison a separate baseline.
- **`replica_count` is 5 but the user did not explicitly ask for a full
  reset.** Stop and use the default count of 3.
- **The source result is noisy or suspicious.** Stop. Replicating one result
  amplifies that measurement into the baseline, so it must be reviewed first.
- **HF sync fails.** Stop for destructive reseeds. A stale or empty sync can
  make the old baseline look missing.
- **Candidate still violates fixed thresholds.** Report that this skill only
  handles the rolling HF baseline; update benchmark JSON thresholds in code
  review if maintainers accept the new absolute limit.
- **The user aborts at either confirmation.** Leave the backup and prepared
  records on disk. Nothing should be uploaded.
- **A bad seed was uploaded.** Use the backup and HF history to identify the
  uploaded file, then remove or supersede it with an explicitly reviewed
  corrective record. Do not silently rewrite unrelated history.

## References

- `.agents/skills/reseed-ssim-references/SKILL.md` — safety pattern for
  intentional baseline replacement.
- `fastvideo/tests/performance/compare_baseline.py` — normalization, rolling
  median comparison, and persistence rules.
- `fastvideo/tests/performance/hf_store.py` — HF sync, record loading,
  `sanitize()`, and `upload_record()`.
- `fastvideo/tests/performance/test_inference_performance.py` — source result
  JSON schema.
- `.buildkite/performance-benchmarks/tests/*.json` — fixed absolute benchmark
  thresholds, separate from rolling baseline comparisons.

## Changelog

| Date | Change |
|------|--------|
| 2026-05-03 | Initial version. Sister workflow to `reseed-ssim-references`, scoped to one performance `(model_id, gpu_type)` baseline seed with backup, confirmation, provenance, and `success=true` upload. |
| 2026-05-03 | Current policy: replicate one approved shifted source result into 3 success records by default, or 5 only when explicitly requested. Add provenance marker for replicated-source reseeds. |
