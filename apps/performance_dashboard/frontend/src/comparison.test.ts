import assert from "node:assert/strict";
import test from "node:test";

import type { CohortCatalogItem, TrendPoint } from "./api.ts";
import {
  comparisonUnavailableReason,
  metricSegments,
  resolveCompatibleCompareSelection,
  resolveCompareSelection,
  stableCohortColor
} from "./comparison.ts";

function cohort(key: string, comparisonKey: string | null, eligible = true): CohortCatalogItem {
  return {
    key,
    schema: eligible ? "v2" : "legacy",
    title: key,
    gpu_key: "gpu:key",
    gpu_label: "1× GPU",
    hardware_label: "Hardware",
    software_label: "Software",
    recipe_label: "Recipe",
    raw_ids: { hardware_profile_id: "hw", software_profile_id: "sw", recipe_fingerprint: "recipe" },
    comparison: {
      eligible,
      key: comparisonKey,
      reason: eligible ? null : "Legacy records are incompatible.",
      metric_schema: "metrics-v1",
      identity: { workload_id: "workload", variant_id: "variant", benchmark_version: "2", recipe_fingerprint: "recipe" }
    },
    model_id: "model",
    gpu_type: "GPU",
    latest_timestamp: "2026-01-01T00:00:00Z",
    latest_baseline_timestamp: null,
    baseline_eligible: false
  };
}

function point(timestamp: string, latency: number | null): TrendPoint {
  const item = cohort("v2:a", "compare:a");
  return {
    timestamp,
    commit_sha: "a".repeat(40),
    success: true,
    run_source: "scheduled_main",
    baseline_eligible: true,
    branch: "main",
    pr_number: "",
    test_scope: "full",
    build_url: "",
    build_id: "",
    job_id: "",
    metrics: { latency },
    cohort: item,
    workload_id: "workload",
    variant_id: "variant",
    benchmark_version: 2,
    recipe_fingerprint: "recipe",
    hardware_profile_id: "hw",
    software_profile_id: "sw"
  };
}

test("comparison selection removes duplicates, stale keys, and entries above the limit", () => {
  assert.deepEqual(resolveCompareSelection(["a", "a", "stale", "b", "c", "d"], ["a", "b", "c", "d"]), [
    "a",
    "b",
    "c"
  ]);
});

test("stable cohort colors depend only on the canonical key", () => {
  assert.equal(stableCohortColor("v2:abc"), stableCohortColor("v2:abc"));
  assert.match(stableCohortColor("v2:abc"), /^hsl\([0-9]+ 68% 38%\)$/);
});

test("compatibility reports filtered, legacy, and mismatched cohorts", () => {
  const anchor = cohort("a", "compare:a");
  assert.match(comparisonUnavailableReason(cohort("missing", "compare:a"), [anchor], new Set(["a"])) ?? "", /No observations/);
  assert.match(comparisonUnavailableReason(cohort("legacy", null, false), [], new Set(["legacy"])) ?? "", /Legacy/);
  assert.match(comparisonUnavailableReason(cohort("other", "compare:b"), [anchor], new Set(["a", "other"])) ?? "", /differs/);
  assert.equal(comparisonUnavailableReason(cohort("same", "compare:a"), [anchor], new Set(["a", "same"])), null);
});

test("comparison URL resolution rejects stale, legacy, filtered, and incompatible cohorts", () => {
  const cohorts = [
    cohort("a", "compare:a"),
    cohort("same", "compare:a"),
    cohort("other", "compare:b"),
    cohort("legacy", null, false)
  ];

  assert.deepEqual(
    resolveCompatibleCompareSelection(
      ["stale", "legacy", "a", "other", "same"],
      cohorts,
      new Set(["a", "same", "other", "legacy"])
    ),
    ["a", "same"]
  );
});

test("missing observations split line segments instead of interpolating across gaps", () => {
  const timestamps = ["2026-01-01", "2026-01-02", "2026-01-03"];
  const segments = metricSegments([point(timestamps[0], 10), point(timestamps[2], 12)], "latency", timestamps);

  assert.deepEqual(segments.map((segment) => segment.map((item) => item.timestamp)), [
    ["2026-01-01"],
    ["2026-01-03"]
  ]);
});
