import type { CohortCatalogItem, TrendPoint } from "./api";

export const MAX_COMPARE_COHORTS = 3;

export function stableCohortColor(key: string) {
  let hash = 0;
  for (const character of key) {
    hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  }
  return `hsl(${hash % 360} 68% 38%)`;
}

export function resolveCompareSelection(requested: readonly string[], validKeys: readonly string[]) {
  const valid = new Set(validKeys);
  return [...new Set(requested)].filter((key) => valid.has(key)).slice(0, MAX_COMPARE_COHORTS);
}

export function resolveCompatibleCompareSelection(
  requested: readonly string[],
  cohorts: readonly CohortCatalogItem[],
  observationKeys: ReadonlySet<string>
) {
  const byKey = new Map(cohorts.map((cohort) => [cohort.key, cohort]));
  const selected: CohortCatalogItem[] = [];
  for (const key of [...new Set(requested)]) {
    const candidate = byKey.get(key);
    if (!candidate || comparisonUnavailableReason(candidate, selected, observationKeys)) {
      continue;
    }
    selected.push(candidate);
    if (selected.length === MAX_COMPARE_COHORTS) {
      break;
    }
  }
  return selected.map((cohort) => cohort.key);
}

export function comparisonUnavailableReason(
  candidate: CohortCatalogItem,
  selected: readonly CohortCatalogItem[],
  observationKeys: ReadonlySet<string>
) {
  if (!observationKeys.has(candidate.key)) {
    return "No observations match the active Source and date filters.";
  }
  if (!candidate.comparison.eligible) {
    return candidate.comparison.reason ?? "This cohort does not declare compatible benchmark metadata.";
  }
  const anchor = selected.find((cohort) => cohort.comparison.eligible);
  if (anchor && anchor.comparison.key !== candidate.comparison.key) {
    return "Workload, benchmark version, metric schema, or recipe differs from the selected comparison.";
  }
  return null;
}

export function metricSegments(points: readonly TrendPoint[], metricKey: string, timestamps: readonly string[]) {
  const pointsByTimestamp = new Map(points.map((point) => [point.timestamp ?? "", point]));
  const segments: TrendPoint[][] = [];
  let current: TrendPoint[] = [];

  for (const timestamp of timestamps) {
    const point = pointsByTimestamp.get(timestamp);
    const value = point?.metrics[metricKey];
    if (point && value !== null && value !== undefined) {
      current.push(point);
    } else if (current.length) {
      segments.push(current);
      current = [];
    }
  }
  if (current.length) {
    segments.push(current);
  }
  return segments;
}
