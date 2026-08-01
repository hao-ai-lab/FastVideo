import type { RunSource } from "./api";

export const ALL_COHORTS = "all";

export type DashboardUrlState = {
  days: number;
  model: string;
  gpu: string;
  cohort: string | null;
  source: "" | RunSource;
};

const RUN_SOURCES = new Set<RunSource>(["scheduled_main", "pr", "local", "unknown"]);

export function readDashboardUrl(url: URL): DashboardUrlState {
  const rawDays = Number(url.searchParams.get("days"));
  const days = Number.isInteger(rawDays) && rawDays >= 1 && rawDays <= 3650 ? rawDays : 90;
  const rawSource = url.searchParams.get("source") as RunSource | null;
  return {
    days,
    model: url.searchParams.get("model") ?? "",
    gpu: url.searchParams.get("gpu") ?? "",
    cohort: url.searchParams.get("cohort"),
    source: rawSource && RUN_SOURCES.has(rawSource) ? rawSource : ""
  };
}

export function writeDashboardUrl(url: URL, state: DashboardUrlState) {
  const next = new URL(url);
  for (const key of ["days", "model", "gpu", "cohort", "source"]) {
    next.searchParams.delete(key);
  }
  if (state.days !== 90) {
    next.searchParams.set("days", String(state.days));
  }
  if (state.model) {
    next.searchParams.set("model", state.model);
  }
  if (state.gpu) {
    next.searchParams.set("gpu", state.gpu);
  }
  if (state.cohort) {
    next.searchParams.set("cohort", state.cohort);
  }
  if (state.source) {
    next.searchParams.set("source", state.source);
  }
  return next;
}

export function resolveCohortSelection(
  current: string | null,
  validKeys: readonly string[],
  defaultCohortKey: string | null
) {
  if (current === ALL_COHORTS) {
    return ALL_COHORTS;
  }
  if (current && validKeys.includes(current)) {
    return current;
  }
  return defaultCohortKey ?? ALL_COHORTS;
}
