export type OverviewSortKey = "status" | "latest" | "latency" | "throughput" | "memory";
export type SortDirection = "asc" | "desc";

type OverviewRecord = {
  status: "pass" | "fail";
  timestamp: string | null;
  metrics: Record<string, { current: number | null }>;
};

function timestampValue(value: string | null) {
  const timestamp = value ? new Date(value).getTime() : Number.NaN;
  return Number.isNaN(timestamp) ? Number.NEGATIVE_INFINITY : timestamp;
}

function metricValue(record: OverviewRecord, key: OverviewSortKey) {
  if (key === "status" || key === "latest") {
    return null;
  }
  return record.metrics[key]?.current ?? null;
}

export function sortCohortOverview<T extends OverviewRecord>(
  rows: readonly T[],
  key: OverviewSortKey,
  direction: SortDirection
) {
  const multiplier = direction === "asc" ? 1 : -1;
  return [...rows].sort((left, right) => {
    if (key === "status") {
      const statusComparison = Number(left.status === "pass") - Number(right.status === "pass");
      if (statusComparison !== 0) {
        return statusComparison * multiplier;
      }
      return timestampValue(right.timestamp) - timestampValue(left.timestamp);
    }
    if (key === "latest") {
      return (timestampValue(left.timestamp) - timestampValue(right.timestamp)) * multiplier;
    }

    const leftValue = metricValue(left, key);
    const rightValue = metricValue(right, key);
    if (leftValue === null && rightValue === null) {
      return timestampValue(right.timestamp) - timestampValue(left.timestamp);
    }
    if (leftValue === null) {
      return 1;
    }
    if (rightValue === null) {
      return -1;
    }
    const comparison = (leftValue - rightValue) * multiplier;
    return comparison || timestampValue(right.timestamp) - timestampValue(left.timestamp);
  });
}
