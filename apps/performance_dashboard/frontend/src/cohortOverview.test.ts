import assert from "node:assert/strict";
import test from "node:test";

import { sortCohortOverview } from "./cohortOverview.ts";

function row(status: "pass" | "fail", timestamp: string, latency: number | null) {
  return {
    status,
    timestamp,
    metrics: {
      latency: { current: latency },
      throughput: { current: latency },
      memory: { current: latency }
    }
  };
}

test("default status ordering prioritizes failures then recent cohorts", () => {
  const rows = [
    row("pass", "2026-01-04T00:00:00Z", 4),
    row("fail", "2026-01-01T00:00:00Z", 1),
    row("fail", "2026-01-03T00:00:00Z", 3)
  ];

  const sorted = sortCohortOverview(rows, "status", "asc");

  assert.deepEqual(sorted.map((item) => item.timestamp), [
    "2026-01-03T00:00:00Z",
    "2026-01-01T00:00:00Z",
    "2026-01-04T00:00:00Z"
  ]);
});

test("missing metrics remain last in either metric sort direction", () => {
  const missing = row("pass", "2026-01-03T00:00:00Z", null);
  const rows = [missing, row("pass", "2026-01-01T00:00:00Z", 1), row("pass", "2026-01-02T00:00:00Z", 2)];

  assert.equal(sortCohortOverview(rows, "latency", "asc").at(-1), missing);
  assert.equal(sortCohortOverview(rows, "latency", "desc").at(-1), missing);
});

test("latest sort handles newest and oldest directions", () => {
  const older = row("pass", "2026-01-01T00:00:00Z", 1);
  const newer = row("pass", "2026-01-02T00:00:00Z", 2);

  assert.equal(sortCohortOverview([older, newer], "latest", "desc")[0], newer);
  assert.equal(sortCohortOverview([older, newer], "latest", "asc")[0], older);
});
