import { useEffect, useMemo, useState } from "react";

import { fetchSummary, fetchTrends, refreshData, SummaryResponse, TrendGroup } from "./api";

const METRIC_KEYS = ["latency", "throughput", "memory", "text_encoder_time_s", "dit_time_s", "vae_decode_time_s"];

function formatNumber(value: number | null | undefined, precision = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return value.toFixed(precision);
}

function shortSha(value: string | null | undefined) {
  return value ? value.slice(0, 7) : "unknown";
}

function formatTime(value: string | null | undefined) {
  if (!value) {
    return "never";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function TrendChart({ group, metricKey }: { group: TrendGroup; metricKey: string }) {
  const points = group.points
    .map((point, index) => ({
      index,
      value: point.metrics[metricKey],
      success: point.success
    }))
    .filter((point) => point.value !== null && point.value !== undefined) as Array<{
      index: number;
      value: number;
      success: boolean;
    }>;

  if (points.length === 0) {
    return <div className="empty-chart">No data</div>;
  }

  const width = 280;
  const height = 96;
  const pad = 12;
  const min = Math.min(...points.map((point) => point.value));
  const max = Math.max(...points.map((point) => point.value));
  const span = max - min || 1;
  const maxIndex = Math.max(...points.map((point) => point.index)) || 1;
  const xy = (point: { index: number; value: number }) => {
    const x = pad + (point.index / maxIndex) * (width - pad * 2);
    const y = height - pad - ((point.value - min) / span) * (height - pad * 2);
    return `${x},${y}`;
  };

  return (
    <svg className="trend-chart" viewBox={`0 0 ${width} ${height}`} role="img">
      <polyline points={points.map(xy).join(" ")} fill="none" stroke="currentColor" strokeWidth="2.2" />
      {points.map((point) => {
        const [cx, cy] = xy(point).split(",");
        return (
          <circle
            key={`${point.index}-${point.value}`}
            cx={cx}
            cy={cy}
            r="3"
            className={point.success ? "point-pass" : "point-fail"}
          />
        );
      })}
    </svg>
  );
}

export default function App() {
  const [days, setDays] = useState(90);
  const [modelFilter, setModelFilter] = useState("");
  const [gpuFilter, setGpuFilter] = useState("");
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [trends, setTrends] = useState<TrendGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [summaryData, trendData] = await Promise.all([
        fetchSummary(days, modelFilter || undefined, gpuFilter || undefined),
        fetchTrends(days, modelFilter || undefined, gpuFilter || undefined)
      ]);
      setSummary(summaryData);
      setTrends(trendData.groups);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function refresh() {
    setRefreshing(true);
    setError(null);
    try {
      await refreshData();
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRefreshing(false);
    }
  }

  useEffect(() => {
    load();
    const interval = window.setInterval(load, 5 * 60 * 1000);
    return () => window.clearInterval(interval);
  }, [days, modelFilter, gpuFilter]);

  const models = useMemo(() => {
    const values = new Set(summary?.rows.map((row) => row.model_id) ?? []);
    trends.forEach((trend) => values.add(trend.model_id));
    return [...values].sort();
  }, [summary, trends]);

  const gpus = useMemo(() => {
    const values = new Set(summary?.rows.map((row) => row.gpu_type) ?? []);
    trends.forEach((trend) => values.add(trend.gpu_type));
    return [...values].sort();
  }, [summary, trends]);

  const latestRows = summary?.rows ?? [];
  const totalRuns = trends.reduce((total, group) => total + group.points.length, 0);
  const sync = summary?.sync;

  return (
    <main className="dashboard">
      <header className="topbar">
        <div>
          <p className="eyebrow">FastVideo CI</p>
          <h1>Performance Dashboard</h1>
        </div>
        <button className="refresh-button" onClick={refresh} disabled={refreshing || loading}>
          {refreshing ? "Refreshing" : "Refresh"}
        </button>
      </header>

      <section className="filters" aria-label="Filters">
        <label>
          Days
          <input
            type="number"
            min="1"
            max="3650"
            value={days}
            onChange={(event) => setDays(Number(event.target.value) || 90)}
          />
        </label>
        <label>
          Model
          <select value={modelFilter} onChange={(event) => setModelFilter(event.target.value)}>
            <option value="">All models</option>
            {models.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        </label>
        <label>
          GPU
          <select value={gpuFilter} onChange={(event) => setGpuFilter(event.target.value)}>
            <option value="">All GPUs</option>
            {gpus.map((gpu) => (
              <option key={gpu} value={gpu}>
                {gpu}
              </option>
            ))}
          </select>
        </label>
      </section>

      {error && <div className="notice error">Failed to load dashboard data: {error}</div>}
      {loading && <div className="notice">Loading performance data</div>}

      <section className="cards" aria-label="Overview">
        <div className="stat">
          <span>Groups</span>
          <strong>{summary?.count ?? 0}</strong>
        </div>
        <div className="stat">
          <span>Failing</span>
          <strong>{summary?.status_counts.fail ?? 0}</strong>
        </div>
        <div className="stat">
          <span>Runs</span>
          <strong>{totalRuns}</strong>
        </div>
        <div className="stat wide">
          <span>Last sync</span>
          <strong>{formatTime(sync?.last_sync_at)}</strong>
          <small>{sync?.repo_id ?? "FastVideo/performance-tracking"}</small>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Latest Status</h2>
          <span>{latestRows.length} model/GPU groups</span>
        </div>
        {latestRows.length === 0 ? (
          <div className="empty">No records match the selected filters.</div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Stored Status</th>
                  <th>Recomputed</th>
                  <th>Model</th>
                  <th>GPU</th>
                  <th>Commit</th>
                  <th>Baseline N</th>
                  <th>Latency</th>
                  <th>Throughput</th>
                  <th>Memory</th>
                  <th>Worst</th>
                </tr>
              </thead>
              <tbody>
                {latestRows.map((row) => (
                  <tr key={`${row.model_id}-${row.gpu_type}`}>
                    <td>
                      <span className={`badge ${row.status}`}>{row.status}</span>
                    </td>
                    <td>
                      <span className={`badge muted ${row.computed_regression_status}`}>
                        {row.computed_regression_status}
                      </span>
                    </td>
                    <td>{row.model_id}</td>
                    <td>{row.gpu_type}</td>
                    <td>{shortSha(row.commit_sha)}</td>
                    <td>{row.baseline_n}</td>
                    <td>{formatNumber(row.metrics.latency?.current, 3)}</td>
                    <td>{formatNumber(row.metrics.throughput?.current, 3)}</td>
                    <td>{formatNumber(row.metrics.memory?.current, 1)}</td>
                    <td>{formatNumber(row.worst_regression_pct, 1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Trends</h2>
          <span>{days} day window</span>
        </div>
        <div className="trend-grid">
          {trends.length === 0 ? (
            <div className="empty full-width">
              No trend records found in the selected time window. Increase the day range or refresh after new CI
              performance records are uploaded.
            </div>
          ) : (
            trends.map((group) =>
              METRIC_KEYS.map((metricKey) => (
                <article className="trend-card" key={`${group.model_id}-${group.gpu_type}-${metricKey}`}>
                  <div>
                    <h3>{summary?.rows[0]?.metrics[metricKey]?.label ?? metricKey}</h3>
                    <p>
                      {group.model_id} | {group.gpu_type}
                    </p>
                  </div>
                  <TrendChart group={group} metricKey={metricKey} />
                </article>
              ))
            )
          )}
        </div>
      </section>
    </main>
  );
}
