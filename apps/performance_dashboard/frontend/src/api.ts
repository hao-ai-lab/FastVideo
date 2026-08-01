export type MetricValue = {
  current: number | null;
  baseline: number | null;
  regression_pct: number | null;
  absolute_delta: number | null;
  threshold_percent: number;
  threshold_absolute: number;
  gated: boolean;
  threshold_exceeded: boolean;
  regressed: boolean;
  label: string;
  lower_is_better: boolean;
  precision: number;
};

export type CohortValue = string | number | null;

export type ComparisonCohort = {
  workload_id: CohortValue;
  variant_id: CohortValue;
  benchmark_version: CohortValue;
  recipe_fingerprint: CohortValue;
  hardware_profile_id: CohortValue;
  software_profile_id: CohortValue;
};

export type CohortDescriptor = {
  key: string;
  schema: "v2" | "legacy" | "invalid_v2";
  title: string;
  gpu_key: string;
  gpu_label: string;
  hardware_label: string;
  software_label: string;
  recipe_label: string;
  raw_ids: {
    hardware_profile_id: string;
    software_profile_id: string;
    recipe_fingerprint: string;
  };
};

export type SummaryRow = {
  model_id: string;
  gpu_type: string;
  timestamp: string | null;
  commit_sha: string | null;
  success: boolean;
  baseline_n: number;
  worst_regression_pct: number | null;
  threshold_exceeded_metrics: string[];
  failing_metrics: string[];
  computed_regression_status: "pass" | "fail";
  status: "pass" | "fail";
  run_source: RunSource;
  baseline_eligible: boolean;
  branch: string;
  pr_number: string;
  test_scope: string;
  build_url: string;
  build_id: string;
  job_id: string;
  metrics: Record<string, MetricValue>;
  cohort: CohortDescriptor;
} & ComparisonCohort;

export type RunSource = "pr" | "local" | "scheduled_main" | "unknown";

export type SummaryResponse = {
  rows: SummaryRow[];
  count: number;
  status_counts: {
    pass: number;
    fail: number;
  };
  filters: {
    days: number | null;
    trend_window_days?: number;
    model_id: string | null;
    gpu_type: string | null;
    gpu_key: string | null;
    cohort_key: string | null;
    run_source: string | null;
  };
  sync: SyncState;
};

export type TrendPoint = {
  timestamp: string | null;
  commit_sha: string | null;
  success: boolean;
  run_source: RunSource;
  baseline_eligible: boolean;
  branch: string;
  pr_number: string;
  test_scope: string;
  build_url: string;
  build_id: string;
  job_id: string;
  metrics: Record<string, number | null>;
  cohort: CohortDescriptor;
} & ComparisonCohort;

export type TrendGroup = {
  model_id: string;
  gpu_type: string;
  points: TrendPoint[];
  cohort: CohortDescriptor;
} & ComparisonCohort;

export type CohortCatalogItem = CohortDescriptor & {
  model_id: string;
  gpu_type: string;
  latest_timestamp: string | null;
  latest_baseline_timestamp: string | null;
  baseline_eligible: boolean;
};

export type CohortCatalogResponse = {
  models: string[];
  gpus: Array<{
    key: string;
    label: string;
    gpu_type: string;
  }>;
  cohorts: CohortCatalogItem[];
  default_cohort_key: string | null;
  sync: SyncState;
};

export type TrendsResponse = {
  groups: TrendGroup[];
  count: number;
  sync: SyncState;
};

export type SyncState = {
  ok: boolean;
  repo_id: string;
  tracking_root: string;
  last_sync_at: string | null;
  last_sync_error: string | null;
};

const jsonHeaders = {
  Accept: "application/json"
};

function params(values: Record<string, string | number | null | undefined>) {
  const out = new URLSearchParams();
  for (const [key, value] of Object.entries(values)) {
    if (value !== null && value !== undefined && value !== "") {
      out.set(key, String(value));
    }
  }
  return out.toString();
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { headers: jsonHeaders });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

type DashboardQuery = {
  days?: number;
  modelId?: string;
  gpuKey?: string;
  cohortKey?: string;
  runSource?: string;
};

export async function fetchCohorts(modelId?: string, gpuKey?: string) {
  return getJson<CohortCatalogResponse>(
    `/api/performance/cohorts?${params({ model_id: modelId, gpu_key: gpuKey })}`
  );
}

export async function fetchSummary({ days = 90, modelId, gpuKey, cohortKey, runSource }: DashboardQuery = {}) {
  return getJson<SummaryResponse>(
    `/api/performance/summary?${params({
      days,
      model_id: modelId,
      gpu_key: gpuKey,
      cohort_key: cohortKey,
      run_source: runSource
    })}`
  );
}

export async function fetchTrends({ days = 90, modelId, gpuKey, cohortKey, runSource }: DashboardQuery = {}) {
  return getJson<TrendsResponse>(
    `/api/performance/trends?${params({
      days,
      model_id: modelId,
      gpu_key: gpuKey,
      cohort_key: cohortKey,
      run_source: runSource
    })}`
  );
}

export async function refreshData() {
  const response = await fetch("/api/performance/refresh", { method: "POST", headers: jsonHeaders });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<SyncState>;
}
