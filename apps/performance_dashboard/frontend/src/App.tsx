import { useEffect, useMemo, useState } from "react";

import { fetchCohorts, fetchSummary, fetchTrends, refreshData } from "./api";
import type {
  CohortCatalogResponse,
  CohortDescriptor,
  RunSource,
  SummaryResponse,
  TrendGroup,
  TrendPoint
} from "./api";
import {
  ALL_COHORTS,
  readDashboardUrl,
  resolveCohortSelection,
  writeDashboardUrl
} from "./dashboardState";

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

function cohortIdentifiers(cohort: CohortDescriptor) {
  const { hardware_profile_id, software_profile_id, recipe_fingerprint } = cohort.raw_ids;
  return (
    [hardware_profile_id, software_profile_id, recipe_fingerprint].filter(Boolean).join(" · ") || "Legacy identity"
  );
}

function CohortLabel({ cohort, compact = false }: { cohort: CohortDescriptor; compact?: boolean }) {
  return (
    <div className={`cohort-cell cohort-${cohort.schema}`}>
      <strong>{cohort.title}</strong>
      <span>{compact ? cohort.gpu_label : `${cohort.gpu_label} · ${cohort.software_label}`}</span>
      {!compact ? <code>{cohortIdentifiers(cohort)}</code> : null}
    </div>
  );
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
  const ariaLabel = `${metricLabel(metricKey)} trend for ${group.model_id}, ${group.cohort.title}, ${
    group.cohort.gpu_label
  }`;

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
          const pointLabel = `${metricLabel(metricKey)} ${formatMetricValue(
            metricKey,
            point.value,
            true
          )} at ${formatTime(point.point.timestamp)}, commit ${shortSha(point.point.commit_sha)}, ${runSourceLabel(
            point.point.run_source
          )}`;
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
        <span>{group.cohort.hardware_label}</span>
        <span>{group.cohort.software_label}</span>
        <code>{cohortIdentifiers(group.cohort)}</code>
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
  const initialUrlState = useMemo(() => readDashboardUrl(new URL(window.location.href)), []);
  const [days, setDays] = useState(initialUrlState.days);
  const [modelFilter, setModelFilter] = useState(initialUrlState.model);
  const [gpuFilter, setGpuFilter] = useState(initialUrlState.gpu);
  const [cohortFilter, setCohortFilter] = useState<string | null>(initialUrlState.cohort);
  const [sourceFilter, setSourceFilter] = useState<"" | RunSource>(initialUrlState.source);
  const [catalog, setCatalog] = useState<CohortCatalogResponse | null>(null);
  const [catalogResolved, setCatalogResolved] = useState(false);
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [trends, setTrends] = useState<TrendGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    setRefreshing(true);
    setError(null);
    try {
      await refreshData();
      setRefreshVersion((version) => version + 1);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRefreshing(false);
    }
  }

  useEffect(() => {
    let cancelled = false;
    setCatalogResolved(false);
    fetchCohorts(modelFilter || undefined, gpuFilter || undefined)
      .then((data) => {
        if (cancelled) {
          return;
        }
        if (modelFilter && !data.models.includes(modelFilter)) {
          setModelFilter("");
          setGpuFilter("");
          return;
        }
        if (gpuFilter && !data.gpus.some((gpu) => gpu.key === gpuFilter)) {
          setGpuFilter("");
          return;
        }
        setCatalog(data);
        setCohortFilter((current) =>
          resolveCohortSelection(current, data.cohorts.map((cohort) => cohort.key), data.default_cohort_key)
        );
        setCatalogResolved(true);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          setCatalogResolved(true);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [modelFilter, gpuFilter, refreshVersion]);

  useEffect(() => {
    if (!catalogResolved || cohortFilter === null) {
      return;
    }
    let cancelled = false;
    const cohortKey = cohortFilter === ALL_COHORTS ? undefined : cohortFilter;
    const query = {
      days,
      modelId: modelFilter || undefined,
      gpuKey: gpuFilter || undefined,
      cohortKey,
      runSource: sourceFilter || undefined
    };
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [summaryData, trendData] = await Promise.all([fetchSummary(query), fetchTrends(query)]);
        if (!cancelled) {
          setSummary(summaryData);
          setTrends(trendData.groups);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }
    load();
    const interval = window.setInterval(load, 5 * 60 * 1000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [catalogResolved, cohortFilter, days, modelFilter, gpuFilter, sourceFilter, refreshVersion]);

  useEffect(() => {
    if (!catalogResolved || cohortFilter === null) {
      return;
    }
    const next = writeDashboardUrl(new URL(window.location.href), {
      days,
      model: modelFilter,
      gpu: gpuFilter,
      cohort: cohortFilter,
      source: sourceFilter
    });
    window.history.replaceState(null, "", `${next.pathname}${next.search}${next.hash}`);
  }, [catalogResolved, cohortFilter, days, modelFilter, gpuFilter, sourceFilter]);

  const latestRows = summary?.rows ?? [];
  const totalRuns = trends.reduce((total, group) => total + group.points.length, 0);
  const sync = summary?.sync;
  const selectedCohort = catalog?.cohorts.find((cohort) => cohort.key === cohortFilter);

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
          <select
            value={modelFilter}
            onChange={(event) => {
              setModelFilter(event.target.value);
              setGpuFilter("");
            }}
          >
            <option value="">All models</option>
            {(catalog?.models ?? []).map((model) => (
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
            {(catalog?.gpus ?? []).map((gpu) => (
              <option key={gpu.key} value={gpu.key}>
                {gpu.label}
              </option>
            ))}
          </select>
        </label>
        <label className="cohort-filter">
          Benchmark cohort
          <select
            value={cohortFilter ?? ""}
            onChange={(event) => setCohortFilter(event.target.value)}
            disabled={!catalogResolved}
          >
            <option value={ALL_COHORTS}>All cohorts</option>
            {(catalog?.cohorts ?? []).map((cohort) => (
              <option key={cohort.key} value={cohort.key}>
                {cohort.schema === "legacy" ? "Legacy · " : ""}
                {cohort.title} — {cohort.gpu_label} — {cohort.software_label} — {cohortIdentifiers(cohort)}
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

      {selectedCohort ? (
        <section className="selected-cohort" aria-label="Selected benchmark cohort">
          <CohortLabel cohort={selectedCohort} />
        </section>
      ) : null}

      {error && <div className="notice error">Failed to load dashboard data: {error}</div>}
      {loading && <div className="notice">Loading performance data</div>}

      <section className="cards" aria-label="Overview">
        <div className="stat">
          <span>Cohorts</span>
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
          <span>{latestRows.length} comparison cohorts</span>
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
                  <th>Cohort</th>
                  <th>Commit</th>
                  <th>Source</th>
                  <th>Schedule / Run type</th>
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
                  <tr key={row.cohort.key}>
                    <td>
                      <span className={`badge ${row.status}`}>{row.status}</span>
                    </td>
                    <td>
                      <span className={`badge muted ${row.computed_regression_status}`}>
                        {row.computed_regression_status}
                      </span>
                    </td>
                    <td>{row.model_id}</td>
                    <td>{row.cohort.gpu_label}</td>
                    <td><CohortLabel cohort={row.cohort} /></td>
                    <td>{shortSha(row.commit_sha)}</td>
                    <td>
                      <span className={`source-badge source-${row.run_source}`}>{runSourceLabel(row.run_source)}</span>
                    </td>
                    <td>{row.test_scope || "Unavailable"}</td>
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
                <article className="trend-card" key={`${group.cohort.key}-${metricKey}`}>
                  <div>
                    <h3>{metricLabel(metricKey)}</h3>
                    <p>{group.model_id}</p>
                    <CohortLabel cohort={group.cohort} compact />
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
