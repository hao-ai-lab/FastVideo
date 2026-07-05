import { useEffect, useMemo, useState } from "react";

import { fetchSummary, fetchTrends, refreshData, RunSource, SummaryResponse, TrendGroup, TrendPoint } from "./api";

const METRIC_KEYS = ["latency", "throughput", "memory", "text_encoder_time_s", "dit_time_s", "vae_decode_time_s"];
const RUN_SOURCES: Array<{ value: "" | RunSource; label: string }> = [
  { value: "", label: "All sources" },
  { value: "scheduled_main", label: "Scheduled main" },
  { value: "pr", label: "PR" },
  { value: "local", label: "Local" },
  { value: "unknown", label: "Unknown" }
];

const METRIC_DEFINITIONS: Record<
  string,
  {
    label: string;
    unit: string;
    precision: number;
    tooltipPrecision: number;
    secondary?: (value: number) => string;
  }
> = {
  latency: {
    label: "Latency",
    unit: "s",
    precision: 2,
    tooltipPrecision: 3,
    secondary: (value) => `${formatNumber(value * 1000, 0)} ms`
  },
  throughput: { label: "Throughput", unit: "FPS", precision: 2, tooltipPrecision: 3 },
  memory: {
    label: "Memory",
    unit: "MB",
    precision: 0,
    tooltipPrecision: 1,
    secondary: (value) => `${formatNumber(value / 1024, 2)} GB`
  },
  text_encoder_time_s: {
    label: "Text Encoder",
    unit: "s",
    precision: 2,
    tooltipPrecision: 3,
    secondary: (value) => `${formatNumber(value * 1000, 0)} ms`
  },
  dit_time_s: {
    label: "DiT",
    unit: "s",
    precision: 2,
    tooltipPrecision: 3,
    secondary: (value) => `${formatNumber(value * 1000, 0)} ms`
  },
  vae_decode_time_s: {
    label: "VAE Decode",
    unit: "s",
    precision: 2,
    tooltipPrecision: 3,
    secondary: (value) => `${formatNumber(value * 1000, 0)} ms`
  }
};

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

function formatDate(value: string | null | undefined) {
  if (!value) {
    return "unknown";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function runSourceLabel(value: string | null | undefined) {
  if (value === "scheduled_main") {
    return "Scheduled main";
  }
  if (value === "pr") {
    return "PR";
  }
  if (value === "local") {
    return "Local";
  }
  return "Unknown";
}

function metricLabel(metricKey: string) {
  return METRIC_DEFINITIONS[metricKey]?.label ?? metricKey;
}

function formatMetricValue(metricKey: string, value: number | null | undefined, tooltip = false) {
  const definition = METRIC_DEFINITIONS[metricKey];
  if (!definition) {
    return formatNumber(value, tooltip ? 3 : 2);
  }
  const formatted = formatNumber(value, tooltip ? definition.tooltipPrecision : definition.precision);
  return formatted === "n/a" ? formatted : `${formatted} ${definition.unit}`;
}

type ChartPoint = {
  plotIndex: number;
  value: number;
  point: TrendPoint;
  x: number;
  y: number;
};

function TrendChart({ group, metricKey }: { group: TrendGroup; metricKey: string }) {
  const [activePoint, setActivePoint] = useState<ChartPoint | null>(null);
  const points = group.points
    .map((point) => ({
      point,
      value: point.metrics[metricKey]
    }))
    .filter((point) => point.value !== null && point.value !== undefined) as Array<{
      point: TrendPoint;
      value: number;
    }>;

  if (points.length === 0) {
    return <div className="empty-chart">No data</div>;
  }

  const width = 360;
  const height = 190;
  const margin = { top: 16, right: 18, bottom: 34, left: 54 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const min = Math.min(...points.map((point) => point.value));
  const max = Math.max(...points.map((point) => point.value));
  const span = max - min || 1;
  const xDenominator = Math.max(points.length - 1, 1);
  const yTicks = [max, min + span / 2, min];
  const chartPoints: ChartPoint[] = points.map((point, plotIndex) => {
    const x = margin.left + (plotIndex / xDenominator) * plotWidth;
    const y = margin.top + (1 - (point.value - min) / span) * plotHeight;
    return { ...point, plotIndex, x, y };
  });
  const rawXTicks = chartPoints.length === 1
    ? [chartPoints[0]]
    : [chartPoints[0], chartPoints[Math.floor((chartPoints.length - 1) / 2)], chartPoints[chartPoints.length - 1]];
  const xTicks = rawXTicks.filter(
    (point, index, items) => items.findIndex((candidate) => candidate.plotIndex === point.plotIndex) === index
  );
  const metric = METRIC_DEFINITIONS[metricKey];
  const selectedPoint = activePoint ?? chartPoints[chartPoints.length - 1];
  const activePointStyle = activePoint
    ? {
        left: `${(activePoint.x / width) * 100}%`,
        top: `${(activePoint.y / height) * 100}%`
      }
    : undefined;
  const ariaLabel = `${metricLabel(metricKey)} trend for ${group.model_id} on ${group.gpu_type}`;

  return (
    <div className="chart-shell">
      <svg className="trend-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={ariaLabel}>
        <line className="axis-line" x1={margin.left} y1={margin.top} x2={margin.left} y2={height - margin.bottom} />
        <line
          className="axis-line"
          x1={margin.left}
          y1={height - margin.bottom}
          x2={width - margin.right}
          y2={height - margin.bottom}
        />
        {yTicks.map((tick) => {
          const y = margin.top + (1 - (tick - min) / span) * plotHeight;
          return (
            <g key={`y-${tick}`}>
              <line className="grid-line" x1={margin.left} y1={y} x2={width - margin.right} y2={y} />
              <text className="axis-label" x={margin.left - 8} y={y + 4} textAnchor="end">
                {formatMetricValue(metricKey, tick)}
              </text>
            </g>
          );
        })}
        {xTicks.map((point) => (
          <text
            className="axis-label"
            key={`x-${point.plotIndex}-${point.point.timestamp ?? ""}`}
            x={point.x}
            y={height - 10}
            textAnchor={point.plotIndex === 0 ? "start" : point.plotIndex === chartPoints.length - 1 ? "end" : "middle"}
          >
            {formatDate(point.point.timestamp)}
          </text>
        ))}
        <polyline
          points={chartPoints.map((point) => `${point.x},${point.y}`).join(" ")}
          fill="none"
          stroke="currentColor"
          strokeWidth="2.2"
        />
        {chartPoints.map((point) => {
          const pointLabel = `${metricLabel(metricKey)} ${formatMetricValue(metricKey, point.value, true)} at ${formatTime(
            point.point.timestamp
          )}, commit ${shortSha(point.point.commit_sha)}, ${runSourceLabel(point.point.run_source)}`;
          return (
            <g
              key={`${point.plotIndex}-${point.value}-${point.point.commit_sha ?? ""}`}
              onMouseEnter={() => setActivePoint(point)}
              onMouseLeave={() => setActivePoint(null)}
            >
              <title>{pointLabel}</title>
              <circle
                className="point-hit-area"
                cx={point.x}
                cy={point.y}
                r="12"
                tabIndex={0}
                aria-label={pointLabel}
                onBlur={() => setActivePoint(null)}
                onFocus={() => setActivePoint(point)}
              />
              <circle
                cx={point.x}
                cy={point.y}
                r={activePoint?.plotIndex === point.plotIndex ? 5 : 4}
                className={point.point.success ? "point-pass point-marker" : "point-fail point-marker"}
              />
            </g>
          );
        })}
      </svg>
      {activePoint ? (
        <div className="hover-tooltip" style={activePointStyle} role="tooltip">
          <strong>{formatMetricValue(metricKey, activePoint.value, true)}</strong>
          {metric?.secondary ? <span>{metric.secondary(activePoint.value)}</span> : null}
          <span>{shortSha(activePoint.point.commit_sha)}</span>
          <span>{runSourceLabel(activePoint.point.run_source)}</span>
        </div>
      ) : null}
      <div className="point-tooltip" aria-live="polite">
        <strong>
          {formatMetricValue(metricKey, selectedPoint.value, true)}
          {metric?.secondary ? <span> ({metric.secondary(selectedPoint.value)})</span> : null}
        </strong>
        <span>{formatTime(selectedPoint.point.timestamp)}</span>
        <span>Commit {shortSha(selectedPoint.point.commit_sha)}</span>
        <span>{runSourceLabel(selectedPoint.point.run_source)}</span>
        <span>{selectedPoint.point.success ? "Stored status: pass" : "Stored status: fail"}</span>
        <span>{selectedPoint.point.baseline_eligible ? "Baseline eligible" : "Not baseline eligible"}</span>
        {selectedPoint.point.pr_number ? <span>PR #{selectedPoint.point.pr_number}</span> : null}
        {selectedPoint.point.branch ? <span>Branch {selectedPoint.point.branch}</span> : null}
        {selectedPoint.point.build_url ? (
          <a href={selectedPoint.point.build_url} target="_blank" rel="noreferrer">
            Buildkite
          </a>
        ) : null}
      </div>
    </div>
  );
}

export default function App() {
  const [days, setDays] = useState(90);
  const [modelFilter, setModelFilter] = useState("");
  const [gpuFilter, setGpuFilter] = useState("");
  const [sourceFilter, setSourceFilter] = useState<"" | RunSource>("");
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
        fetchSummary(days, modelFilter || undefined, gpuFilter || undefined, sourceFilter || undefined),
        fetchTrends(days, modelFilter || undefined, gpuFilter || undefined, sourceFilter || undefined)
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
  }, [days, modelFilter, gpuFilter, sourceFilter]);

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
        <label>
          Source
          <select value={sourceFilter} onChange={(event) => setSourceFilter(event.target.value as "" | RunSource)}>
            {RUN_SOURCES.map((source) => (
              <option key={source.value || "all"} value={source.value}>
                {source.label}
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
                  <th>Source</th>
                  <th>Baseline</th>
                  <th>Baseline N</th>
                  <th>Latency</th>
                  <th>Throughput</th>
                  <th>Memory</th>
                  <th>Worst</th>
                  <th>Exceeded</th>
                  <th>Failing</th>
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
                    <td>
                      <span className={`source-badge source-${row.run_source}`}>{runSourceLabel(row.run_source)}</span>
                    </td>
                    <td>{row.baseline_eligible ? "eligible" : "excluded"}</td>
                    <td>{row.baseline_n}</td>
                    <td>{formatNumber(row.metrics.latency?.current, 3)}</td>
                    <td>{formatNumber(row.metrics.throughput?.current, 3)}</td>
                    <td>{formatNumber(row.metrics.memory?.current, 1)}</td>
                    <td>{formatNumber(row.worst_regression_pct, 1)}%</td>
                    <td>
                      {row.threshold_exceeded_metrics.length
                        ? row.threshold_exceeded_metrics.join(", ")
                        : "none"}
                    </td>
                    <td>{row.failing_metrics.length ? row.failing_metrics.join(", ") : "none"}</td>
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
                    <h3>{metricLabel(metricKey)}</h3>
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
