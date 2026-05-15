# Performance Redesign Commit Stack

  ## Summary

  Break the redesign into a stacked sequence where each commit is reviewable and should keep the
  performance job runnable. The stack starts with docs/schema/config, then turns on v2 result emission,
  storage, comparison# Performance Redesign Commit Stack

  ## Summary

  Break the redesign into a stacked sequence where each commit is reviewable and keeps the performance job
  runnable. The stack starts with docs/schema/config, then adds v2 result emission, identity-scoped
  storage, exact comparison, dashboarding, and deeper timing/rank instrumentation.

  Important correction: CALIBRATION_NEEDED records should not automatically seed baselines. New hardware/
  software/runtime cohorts are seeded manually through the reseed-performance workflow after review.

  ## Commit Stack

  1. [docs]: add performance redesign plan
      - Commit fastvideo/tests/performance/redesign_plan.md.
      - No code behavior changes.
  2. [perf]: add v2 performance schema and identity helpers
      - Add helpers for statuses, metric policies, stable JSON hashing, recipe fingerprinting, hardware/
        software profile IDs, and v1/v2 field flattening.
      - Add unit tests for stable fingerprints, changed recipe fields, and software profile changes.
  3. [perf]: version benchmark configs with workload and variant identity
      - Update .buildkite/performance-benchmarks/tests/wan-t2v-1.3b.json with workload_id,
        variant_id=canonical, benchmark_version=1, comparison policy, and quality policy.
      - Keep benchmark_id and existing thresholds for compatibility.
  4. [perf]: emit v2 benchmark results with legacy compatibility
      - Update test_inference_performance.py to write v2 fields: identity, provenance, recipe, hardware,
        software, metrics, runs, and quality status.
      - Preserve existing flat fields so old comparison paths still work during the stack.
      - Reset CUDA peak memory before each measured run and store median metrics in v2.
  5. [perf]: support identity-scoped HF storage with legacy reads
      - Update hf_store.py to write v2 records under <workload>/<variant>/<hardware>/<software>/....
      - Keep legacy <model_id>/... reads and dataframe loading.
      - Add load_records_for_identity(...) while leaving load_records_for_model(...) intact.
  6. [perf]: compare v2 records by exact benchmark identity
      - Update compare_baseline.py to prefer v2 identity comparison and fall back to v1 only for legacy
        records.
      - Implement statuses: PASS, REGRESSION, CALIBRATION_NEEDED, RECIPE_MISMATCH, INFRA_ERROR,
        QUALITY_BLOCKED.
      - Add metric-specific percent and absolute thresholds.
      - Missing exact baseline returns CALIBRATION_NEEDED and does not persist as a successful baseline
        seed.
      - Only manually reseeded/promoted records are used to initialize a new comparable baseline.
  7. [perf]: make scheduled-main provenance and persistence explicit
      - Add scheduled-main provenance fields: trigger_type, schedule_id, build_id, build_url, branch,
        commit_sha.
      - Gate persistence with explicit scheduled-main env support while keeping current TEST_SCOPE=full &&
        branch=main compatibility.
      - Remove required PR provenance; allow optional ad-hoc change_request.
  8. [perf]: update dashboard for v2 identities and statuses
      - Group dashboard charts by workload, variant, hardware profile, software profile, and benchmark
        version.
      - Show status, quality status, recipe fingerprint, software profile, and legacy v1 charts separately.
      - Make calibration-needed cohorts visible but distinct from accepted baselines.
  9. [perf]: collect low-overhead denoising timing metrics
      - Extend PipelineLoggingInfo and DenoisingStage instrumentation for denoise step, transformer
        forward, and scheduler step timings.
      - Prefer low-overhead CUDA-event timing when available; keep behavior disabled unless performance/
        stage logging is enabled.
      - Map these into v2 metrics when present.
  10. [perf]: aggregate rank-level distributed performance metrics

  - Update multiprocess executor responses to include rank-level timing and memory metadata.
  - Store rank metrics under v2 runs; compare rank-max memory/timing where applicable.
  - Preserve rank-0 legacy fields for compatibility.

  11. [perf]: enforce candidate quality metadata

  - Implement candidate/quality handling for variants.
  - Candidate variants can collect metrics but are not promotable when required quality evidence is
    missing.
  - Update the performance reseed skill/workflow to target one exact v2 identity tuple.
  - Require provenance for runtime upgrades and explicit marking of accepted baseline records.
  - Manual reseed writes the records that make a new cohort comparable.
  - Keep old v1 reseed notes only as legacy guidance.
  - After commits 2-6: run unit tests for schema, HF storage, and comparison logic.
  - After commit 4: run a local import/discovery check for benchmark configs and helper serialization.
  - After commit 6: test v1 legacy records, v2 missing baseline, recipe mismatch, regression, and pass
    paths.
  - After commit 8: test dashboard grouping with synthetic v1 and v2 records.
    scoped.
  - Before merging the full stack: run the Modal performance job once in non-persistent/dry-run mode.

  ## Assumptions

  - This is a stacked commit sequence; each commit is independently reviewable but may depend on previous
    commits.
  - v2 records are not compared to v1 records.
  - CALIBRATION_NEEDED exits successfully by default but is visible in summaries.
  - Scheduled-main calibration records are audit data, not automatic baseline seeds.
  - New cohorts become comparable only after manual reseed/promotion.
  - Promoted baseline support is optional until records are explicitly marked accepted.
