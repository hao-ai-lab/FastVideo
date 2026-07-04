export type MetricValue = {
  current: number | null;
  baseline: number | null;
  regression_pct: number | null;
  absolute_delta: number | null;
  threshold_percent: number;
  threshold_absolute: number;
  gated: boolean;
  regressed: boolean;
  label: string;
  lower_is_better: boolean;
  precision: number;
};

export type SummaryRow = {
  model_id: string;
  gpu_type: string;
  timestamp: string | null;
  commit_sha: string | null;
  success: boolean;
  baseline_n: number;
  worst_regression_pct: number | null;
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
};

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
};

export type TrendGroup = {
  model_id: string;
  gpu_type: string;
  points: TrendPoint[];
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

export async function fetchSummary(days = 90, modelId?: string, gpuType?: string, runSource?: string) {
  return getJson<SummaryResponse>(
    `/api/performance/summary?${params({ days, model_id: modelId, gpu_type: gpuType, run_source: runSource })}`
  );
}

export async function fetchTrends(days = 90, modelId?: string, gpuType?: string, runSource?: string) {
  return getJson<TrendsResponse>(
    `/api/performance/trends?${params({ days, model_id: modelId, gpu_type: gpuType, run_source: runSource })}`
  );
}

export async function refreshData() {
  const response = await fetch("/api/performance/refresh", { method: "POST", headers: jsonHeaders });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<SyncState>;
}
