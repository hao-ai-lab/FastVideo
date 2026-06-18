export type MetricValue = {
  current: number | null;
  baseline: number | null;
  regression_pct: number | null;
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
  regression_threshold_pct: number;
  computed_regression_status: "pass" | "fail";
  status: "pass" | "fail";
  metrics: Record<string, MetricValue>;
};

export type SummaryResponse = {
  rows: SummaryRow[];
  count: number;
  status_counts: {
    pass: number;
    fail: number;
  };
  filters: {
    days: number;
    model_id: string | null;
    gpu_type: string | null;
  };
  sync: SyncState;
};

export type TrendPoint = {
  timestamp: string | null;
  commit_sha: string | null;
  success: boolean;
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

export async function fetchSummary(days = 90, modelId?: string, gpuType?: string) {
  return getJson<SummaryResponse>(
    `/api/performance/summary?${params({ days, model_id: modelId, gpu_type: gpuType })}`
  );
}

export async function fetchTrends(days = 90, modelId?: string, gpuType?: string) {
  return getJson<TrendsResponse>(
    `/api/performance/trends?${params({ days, model_id: modelId, gpu_type: gpuType })}`
  );
}

export async function refreshData() {
  const response = await fetch("/api/performance/refresh", { method: "POST", headers: jsonHeaders });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<SyncState>;
}
