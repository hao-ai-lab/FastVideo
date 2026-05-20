---
name: reseed-performance-baseline
description: Re-seed the HF performance-tracking baseline for an intentional runtime, dependency, or environment-caused benchmark shift using one or more reviewed normalized performance JSONs. Use when performance CI fails because metrics such as latency, throughput, component time, or peak memory changed for an accepted reason and the rolling median baseline in FastVideo/performance-tracking must be advanced from a consistent batch of reviewed source results. The workflow backs up existing history under /tmp, validates all source JSONs for the same (model_id, gpu_type), rejects internally inconsistent source batches, uploads one success=true reseed record per accepted source JSON, and offers to clean local temp state after a successful upload.
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

This skill now reseeds from a reviewed batch of one or more source performance
JSONs. It uploads one new `success=true` record per accepted source JSON; it
does not blindly replicate one measurement into 3 or 5 records. The effective
reseed size is therefore dynamic and equals the number of provided, validated,
internally consistent source JSONs.

If the operator provides fewer than 3 records, call out that the last-5 rolling
median may not move immediately. If the operator provides 3 consistent shifted
records, the rolling median usually moves immediately. If the operator provides
5 consistent shifted records, the last-5 window is effectively reset to the new
runtime profile.

These records are intentional operator-approved baseline resets, not ordinary
independent main-branch persistence. Mark them clearly with provenance fields
so the HF history remains auditable.

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
- One or more shifted source result JSONs have been reviewed and accepted, and
  the operator wants to use those exact reviewed results to advance the rolling
  baseline.
- The source batch is internally consistent: no provided source JSON regresses
  against the batch median by more than the configured tolerance.

## When not to use

- The benchmark failure might be a real code regression. Fix or investigate
  the code path first.
- The fixed benchmark thresholds in
  `.buildkite/performance-benchmarks/tests/*.json` are too low. Those are a
  separate gate from the rolling HF baseline and may need a code review change.
- There is no clear source run, commit, and rationale. Baseline history is a
  production signal; do not edit it without provenance.
- The provided source JSONs disagree materially with each other. Rerun or
  investigate instead of uploading a noisy reseed batch.

## Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `model_id` | Yes | Benchmark id, e.g. `wan-t2v-1.3b-2gpu`. This maps to the HF subdirectory after `sanitize(model_id)`. |
| `gpu_type` | Yes | Exact GPU device string from the performance record, e.g. the L40S device name emitted by CI. Baselines are GPU-specific. |
| `source_results` | Yes | One or more local paths or Buildkite artifact URLs for accepted shifted performance JSONs. Prefer normalized `normalized_perf_*.json` artifacts emitted by `compare_baseline.py`. Accept `source_result` as an alias only for a single JSON. |
| `max_intra_batch_regression` | No | Maximum allowed regression of any source JSON against the source batch median. Default: `PERF_MAX_REGRESSION` if set, otherwise `0.05` (5%). |
| `intent_rationale` | Yes | One-line explanation for why the baseline shift is legitimate. This is written into provenance and should be reused in the PR. |

Hardcoded defaults:

- HF repo: `FastVideo/performance-tracking` (`HF_REPO_ID` override is
  supported by the code, but use the default unless the user explicitly asks).
- Local sync root: `/tmp/perf-tracking` (`PERFORMANCE_TRACKING_ROOT` override
  is supported).
- Backup root: `/tmp/performance_reseed_backup`.
- Download scratch root for source artifact URLs: `/tmp/performance_reseed_source`.
- Baseline window: last 5 `success=true` records for the same
  `(model_id, gpu_type)`.
- Reseed count: dynamic. Upload exactly one accepted seed record per validated
  source JSON.

## Steps

### 1. Validate the target and source results

Normalize `source_results` to a list. If the user passes a single
`source_result`, treat it as a one-element `source_results` list and report
that a single record may not move the last-5 median immediately.

If any source result is a Buildkite artifact URL, download it first into a
local scratch directory under `/tmp/performance_reseed_source/` and use the
downloaded JSON path for the rest of the workflow. If the agent cannot access
the artifact because Buildkite authentication is missing, ask the user to
download the artifact manually and provide the local path.

Prefer the normalized Buildkite artifact emitted by `compare_baseline.py`:

```text
perf_reports/results/normalized_perf_*.json
```

That file is already in the HF tracking schema. Load each normalized JSON
directly:

```python
import json

with open(source_result, encoding="utf-8") as f:
    record = json.load(f)
```

Stop if any normalized record's `model_id` or `gpu_type` does not match the
requested `model_id` and `gpu_type`.

The source records may have `success: false` when they came from failed
rolling baseline comparisons. That is expected; only the reviewed reseed
records become new `success: true` baseline records after explicit approval.

Sort validated source records by their original `timestamp` ascending before
preparing the seed records. If a source timestamp is missing or unparsable,
preserve input order for those records and print a warning. This makes the
fresh reseed timestamps deterministic and makes it clear which records enter
the last-5 window when more than 5 source JSONs are provided.

Check that `HF_API_KEY` is exported. The sync path may be public, but the
upload path requires write access.

### 1a. Check source batch consistency

Before syncing or preparing uploads, reject source batches that are internally
inconsistent. Use the same metric direction as `compare_baseline.py`:

- Lower is better: `latency`, `memory`, `text_encoder_time_s`, `dit_time_s`,
  `vae_decode_time_s`.
- Higher is better: `throughput`.

For each metric with at least two non-null source values:

1. Compute the source batch median.
2. For lower-is-better metrics, compute `(source_value - batch_median) / batch_median`.
3. For `throughput`, compute `(batch_median - source_value) / batch_median`.
4. Stop if any source record regresses against the batch median by more than
   `max_intra_batch_regression`.

Default `max_intra_batch_regression` to `PERF_MAX_REGRESSION` when set,
otherwise `0.05`. Print a table with per-source values, batch median, and
worst intra-batch regression.

This check prevents uploading a mixed batch where one JSON is materially
slower or faster than the others. If the batch fails this check, ask the user
to provide a cleaner batch or explicitly investigate the variance. Do not
silently drop outliers unless the user gives a concrete reviewed reason and a
new source list.

### 1b. How to obtain source results from CI

The performance CI exports normalized source results for failed rolling
baseline comparisons when `compare_baseline.py` ran. The preferred artifacts
come from:

```text
perf_reports/results/normalized_perf_*.json
```

The normal operator flow is:

1. Open the failed Buildkite performance job or several reruns of the same
   benchmark after the accepted environment shift.
2. Download the `normalized_perf_*.json` artifacts for the target benchmark.
3. Pass all reviewed local paths or artifact URLs as `source_results`.

Do not scrape the Markdown performance summary to reconstruct JSON. The
normalized JSON artifacts are the only supported source of truth for reseed
metrics and provenance. Raw `fastvideo/tests/performance/results/perf_*.json`
artifacts are not accepted by this skill. If no normalized JSON artifact is
present, that run is not a valid source for baseline reseeding.

### 2. Sync and back up existing HF records under /tmp

Use `fastvideo/tests/performance/hf_store.py` helpers directly. Do **not** use
`compare_baseline.py` as a sync shortcut; on full main runs it can persist
records, while this step must only fetch and back up existing history.

The sync command pattern is:

```bash
export PERFORMANCE_TRACKING_ROOT="${PERFORMANCE_TRACKING_ROOT:-/tmp/perf-tracking}"
export HF_REPO_ID="${HF_REPO_ID:-FastVideo/performance-tracking}"
PYTHONPATH=fastvideo/tests/performance python -c 'from hf_store import sync_from_hf; import os; sync_from_hf(os.environ["PERFORMANCE_TRACKING_ROOT"], strict=True)'
```

Then back up only the sanitized model directory under `/tmp`:

```bash
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
MODEL_SAFE=$(PYTHONPATH=fastvideo/tests/performance python - <<'PY'
from hf_store import sanitize
print(sanitize("<model_id>"))
PY
)
BACKUP_DIR="/tmp/performance_reseed_backup/${TIMESTAMP}_${SHORT_COMMIT}_${MODEL_SAFE}"
mkdir -p "$BACKUP_DIR"
cp -R "${PERFORMANCE_TRACKING_ROOT}/${MODEL_SAFE}" "$BACKUP_DIR/" 2>/dev/null || true
```

Write provenance next to the backup:

```bash
cat > "$BACKUP_DIR/PROVENANCE.txt" <<EOF
model_id: <model_id>
gpu_type: <gpu_type>
source_results:
  - <source_result_1>
  - <source_result_2>
reseed_record_count: <len(source_results)>
max_intra_batch_regression: <threshold>
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
from hf_store import load_records_for_model

records = load_records_for_model(
    "/tmp/perf-tracking",
    "<model_id>",
    "<gpu_type>",
    last_n=5,
    successful_only=True,
)
```

Print a small table showing old medians, source batch medians, candidate
medians after appending the proposed seed records, and source batch spread for:

- `latency`
- `throughput`
- `memory`
- `text_encoder_time_s`
- `dit_time_s`
- `vae_decode_time_s`

Also print how many successful old records exist. Make clear:

- 1 seed record usually does not move a last-5 median by itself.
- 3 consistent seed records usually move the last-5 median immediately.
- 5 consistent seed records effectively reset the last-5 window.
- The records are intentional approved baseline resets and must be labeled
  that way.

### 4. Confirm intent

Require an explicit confirmation phrase before preparing the upload:

> About to RE-SEED performance baseline for `<model_id>` on `<gpu_type>`.
> This will upload `<N>` new `success=true` records to
> `FastVideo/performance-tracking/<sanitize(model_id)>/`, one per accepted
> source JSON.
>
> Reason: `<intent_rationale>`
> Source results: `<source_results>`
> Reseed record count: `<N>`
> Max intra-batch regression: `<threshold>`
> Note: these records come from a reviewed source batch and are intended to
> move the rolling median to the accepted runtime profile. They are not
> ordinary main-branch persistence.
> HEAD: `<git rev-parse --short=12 HEAD>`
> Backup: `<BACKUP_DIR>`
>
> Reply `confirm performance reseed` to proceed, anything else to abort.

Do not continue unless the user types exactly `confirm performance reseed`.

### 5. Create the accepted seed records

Create one seed record from each normalized source result. Do not copy the
source JSON wholesale.

Infer the baseline field allowlist from all existing HF records for the target
`(model_id, gpu_type)` after syncing, including both `success=true` and
`success=false` records. Use the union of non-provenance keys present in those
target records, preserving only fields that also exist in the normalized
source record or are explicitly set by the reseed workflow. Always include
`model_id`, `timestamp`, and `success` because the upload path and baseline
loader depend on them. Always set `timestamp` to a fresh reseed timestamp and
`success` to `true`. Do not include unrelated source-only fields that are
absent from existing HF records.

Exclude existing provenance or operator metadata from the inferred baseline
field allowlist. At minimum, exclude keys prefixed with `baseline_reseed` and
any fields known to be local-only audit metadata.

If there are no previous HF records for the target model/GPU, fall back to this
default baseline field list:

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
- `success`

Do not upload extra fields from the source artifact.

Optional provenance fields are allowed and useful:

- `baseline_reseed: true`
- `baseline_reseed_reason`
- `baseline_reseed_source_result`
- `baseline_reseed_source_timestamp`
- `baseline_reseed_source_success`
- `baseline_reseed_batch_size`
- `baseline_reseed_batch_index`
- `baseline_reseed_operator`
- `baseline_reseed_max_intra_batch_regression`

Use a fresh reseed timestamp for each seed record, not the original source
result timestamp. This is required because
`load_records_for_model(..., last_n=5)` keeps the last records after loading
the model directory; stale filenames/timestamps may not enter the last-5
window and therefore may not move the median. Preserve the original source
timestamp in `baseline_reseed_source_timestamp`.

Use the existing filename convention from `_write_tracking_record()`:
`<sanitize(timestamp)>_<sanitize(commit_sha)>.json` under the sanitized model
directory, but include a deterministic suffix such as `_reseed_01`,
`_reseed_02`, and so on before `.json` so multiple records from the same
batch do not overwrite each other.

If a source record already exists on HF with `success=false`, do not edit it
in place unless the user explicitly asked for an audit-preserving correction.
Prefer uploading new accepted seed records so failed history remains visible.

### 6. Pause before upload

Print:

- Backup directory path under `/tmp`.
- Prepared local record paths under `PERFORMANCE_TRACKING_ROOT`.
- HF paths that will receive the new records.
- Old rolling medians.
- Source batch medians, source batch spread, reseed count, and candidate
  medians.
- Rationale.

Ask the user to reply exactly `upload`. Anything else aborts and leaves the
prepared records plus backup on disk.

### 7. Upload only the scoped records

Use the shared storage helper so the path and repo type match CI:

```python
from hf_store import upload_record

upload_record("<local_record_path>", record, strict=True)
```

Run it once per prepared record. Each upload goes to:

```text
FastVideo/performance-tracking/<sanitize(model_id)>/<record_filename>.json
```

Never bulk upload the whole tracking root. Never modify another model's
directory in the same operation.

### 8. Report outcome and offer cleanup

Report:

- Uploaded HF paths.
- Backup directory under `/tmp`.
- Local tracking root, usually `/tmp/perf-tracking`.
- Old baseline window count and medians.
- Source batch medians, source batch spread, reseed count, and candidate
  medians.
- Expected effect based on reseed count.
- Any separate threshold changes still needed in
  `.buildkite/performance-benchmarks/tests/*.json`.

Include the `intent_rationale` in the PR or follow-up comment so reviewers can
distinguish an accepted baseline shift from a hidden regression.

After the upload is verified, ask whether the user wants to clear temporary
local state. Explain what each directory is for:

- `PERFORMANCE_TRACKING_ROOT`, usually `/tmp/perf-tracking`: local synced
  mirror of `FastVideo/performance-tracking` plus the prepared local seed
  records used for scoped upload.
- `/tmp/performance_reseed_backup/<...>`: local backup of the target model's
  pre-reseed HF history plus `PROVENANCE.txt`, kept so a bad reseed can be
  audited or corrected.
- `/tmp/performance_reseed_source/<...>` when used: downloaded source JSON
  artifacts from Buildkite URLs.

Ask:

> Reseed succeeded. Do you want me to delete the local temp tracking mirror,
> source downloads, and reseed backup under `/tmp`? These files are local
> safety/audit artifacts only; HF already has the uploaded records.
>
> Reply `cleanup reseed temp` to delete them, anything else to keep them.

Do not delete anything unless the user replies exactly
`cleanup reseed temp`. If cleanup is requested, remove only the specific
directories created for this reseed. Never remove unrelated `/tmp` contents.

## Failure modes and handling

- **`HF_API_KEY` unset.** Stop before upload. Do not create an untracked
  process that appears to have reseeded but never reached HF.
- **Source result does not match target.** Stop. The wrong benchmark or GPU
  would poison a separate baseline.
- **Source batch is internally inconsistent.** Stop if any source regresses
  against the source batch median by more than `max_intra_batch_regression`.
  Ask for cleaner sources or a reviewed explanation before continuing.
- **Too few source records to move the median.** Continue only after making
  clear that one or two records may not immediately move the last-5 median.
- **The source results are noisy or suspicious.** Stop. Reseeding amplifies
  those measurements into the baseline, so they must be reviewed first.
- **HF sync fails.** Stop for destructive reseeds. A stale or empty sync can
  make the old baseline look missing.
- **Candidate still violates fixed thresholds.** Report that this skill only
  handles the rolling HF baseline; update benchmark JSON thresholds in code
  review if maintainers accept the new absolute limit.
- **The user aborts at either confirmation.** Leave the backup and prepared
  records on disk. Nothing should be uploaded.
- **The user declines cleanup.** Keep `/tmp/perf-tracking`, the source
  download directory if any, and `/tmp/performance_reseed_backup/<...>` in
  place for audit/debugging.
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
| 2026-05-03 | Previous policy: replicate one approved shifted source result into 3 success records by default, or 5 only when explicitly requested. Add provenance marker for replicated-source reseeds. Superseded by the 2026-05-08 dynamic multi-source policy. |
| 2026-05-08 | Replace fixed 3/5 replication with dynamic multi-source reseeding: upload one seed record per reviewed source JSON, validate intra-batch consistency, move backup/source scratch under `/tmp`, and ask whether to clean temp state after successful upload. |
