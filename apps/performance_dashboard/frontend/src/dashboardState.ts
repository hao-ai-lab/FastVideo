import type { RunSource } from "./api";

export const ALL_COHORTS = "all";
const MAX_URL_COMPARE_COHORTS = 3;

export type DashboardUrlState = {
  days: number;
  model: string;
  gpu: string;
  cohort: string | null;
  source: "" | RunSource;
  hardware: string;
  software: string;
  recipe: string;
  compareMode: boolean;
  compareCohorts: string[];
};

const RUN_SOURCES = new Set<RunSource>(["scheduled_main", "pr", "local", "unknown"]);

export function readDashboardUrl(url: URL): DashboardUrlState {
  const rawDays = Number(url.searchParams.get("days"));
  const days = Number.isInteger(rawDays) && rawDays >= 1 && rawDays <= 3650 ? rawDays : 90;
  const rawSource = url.searchParams.get("source") as RunSource | null;
  const compareCohorts = [...new Set((url.searchParams.get("compare") ?? "").split(",").filter(Boolean))].slice(
    0,
    MAX_URL_COMPARE_COHORTS
  );
  return {
    days,
    model: url.searchParams.get("model") ?? "",
    gpu: url.searchParams.get("gpu") ?? "",
    cohort: url.searchParams.get("cohort"),
    source: rawSource && RUN_SOURCES.has(rawSource) ? rawSource : "",
    hardware: url.searchParams.get("hardware") ?? "",
    software: url.searchParams.get("software") ?? "",
    recipe: url.searchParams.get("recipe") ?? "",
    compareMode: url.searchParams.get("mode") === "compare",
    compareCohorts
  };
}

export function writeDashboardUrl(url: URL, state: DashboardUrlState) {
  const next = new URL(url);
  for (const key of [
    "days",
    "model",
    "gpu",
    "cohort",
    "source",
    "hardware",
    "software",
    "recipe",
    "mode",
    "compare"
  ]) {
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
  if (state.hardware) {
    next.searchParams.set("hardware", state.hardware);
  }
  if (state.software) {
    next.searchParams.set("software", state.software);
  }
  if (state.recipe) {
    next.searchParams.set("recipe", state.recipe);
  }
  if (state.compareMode) {
    next.searchParams.set("mode", "compare");
    if (state.compareCohorts.length) {
      next.searchParams.set("compare", [...new Set(state.compareCohorts)].slice(0, MAX_URL_COMPARE_COHORTS).join(","));
    }
  }
  return next;
}

export function resolveAdvancedFilterSelection(current: string, validValues: readonly string[]) {
  return current && validValues.includes(current) ? current : "";
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
