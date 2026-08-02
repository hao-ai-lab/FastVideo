import { useEffect, useMemo, useState } from "react";

import { fetchCohorts, fetchSummary, fetchTrends, refreshData } from "./api";
import type {
  AdvancedFilterOption,
  CohortCatalogResponse,
  CohortCatalogItem,
  CohortDescriptor,
  RunSource,
  SummaryRow,
  SummaryResponse,
  TrendGroup,
  TrendPoint
} from "./api";
import { sortCohortOverview } from "./cohortOverview";
import type { OverviewSortKey, SortDirection } from "./cohortOverview";
import {
  comparisonUnavailableReason,
  MAX_COMPARE_COHORTS,
  metricSegments,
  resolveCompatibleCompareSelection,
  stableCohortColor
} from "./comparison";
import {
  ALL_COHORTS,
  readDashboardUrl,
  resolveAdvancedFilterSelection,
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

function advancedOptionLabel(option: AdvancedFilterOption) {
  return option.raw_id ? `${option.label} — ${option.raw_id}` : `${option.label} — no raw ID`;
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

function formatOverviewMetric(metricKey: string, value: number | null | undefined) {
  return value === null || value === undefined ? "Unavailable" : formatMetricValue(metricKey, value);
}

function SortHeader({
  label,
  sortKey,
  activeKey,
  direction,
  onChange
}: {
  label: string;
  sortKey: OverviewSortKey;
  activeKey: OverviewSortKey;
  direction: SortDirection;
  onChange: (key: OverviewSortKey) => void;
}) {
  const active = sortKey === activeKey;
  return (
    <button className="sort-button" type="button" onClick={() => onChange(sortKey)}>
      {label}
      {active ? <span aria-hidden="true">{direction === "asc" ? " ↑" : " ↓"}</span> : null}
    </button>
  );
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

type ComparePoint = {
  group: TrendGroup;
  point: TrendPoint;
  value: number;
  x: number;
  y: number;
};

function CompareChart({ groups, metricKey }: { groups: TrendGroup[]; metricKey: string }) {
  const [activePoint, setActivePoint] = useState<ComparePoint | null>(null);
  const timestamps = [...new Set(groups.flatMap((group) => group.points.map((point) => point.timestamp).filter(Boolean)))]
    .sort() as string[];
  const values = groups.flatMap((group) =>
    group.points
      .map((point) => point.metrics[metricKey])
      .filter((value): value is number => value !== null && value !== undefined)
  );

  if (!timestamps.length || !values.length) {
    return <div className="empty-chart">Metric unavailable for every selected cohort.</div>;
  }

  const width = 760;
  const height = 280;
  const margin = { top: 18, right: 22, bottom: 42, left: 68 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const xDenominator = Math.max(timestamps.length - 1, 1);
  const xFor = (timestamp: string) => margin.left + (timestamps.indexOf(timestamp) / xDenominator) * plotWidth;
  const yFor = (value: number) => margin.top + (1 - (value - min) / span) * plotHeight;
  const yTicks = [max, min + span / 2, min];
  const rawXTicks = timestamps.length === 1
    ? [timestamps[0]]
    : [timestamps[0], timestamps[Math.floor((timestamps.length - 1) / 2)], timestamps[timestamps.length - 1]];
  const xTicks = [...new Set(rawXTicks)];
  const unavailableGroups = groups.filter((group) =>
    group.points.every((point) => point.metrics[metricKey] === null || point.metrics[metricKey] === undefined)
  );

  return (
    <div className="compare-chart-shell">
      <svg
        className="compare-chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${metricLabel(metricKey)} comparison for ${groups.length} cohorts`}
      >
        <line className="axis-line" x1={margin.left} y1={margin.top} x2={margin.left} y2={height - margin.bottom} />
        <line
          className="axis-line"
          x1={margin.left}
          y1={height - margin.bottom}
          x2={width - margin.right}
          y2={height - margin.bottom}
        />
        {yTicks.map((tick) => {
          const y = yFor(tick);
          return (
            <g key={`y-${tick}`}>
              <line className="grid-line" x1={margin.left} y1={y} x2={width - margin.right} y2={y} />
              <text className="axis-label" x={margin.left - 8} y={y + 4} textAnchor="end">
                {formatMetricValue(metricKey, tick)}
              </text>
            </g>
          );
        })}
        {xTicks.map((timestamp, index) => (
          <text
            className="axis-label"
            key={timestamp}
            x={xFor(timestamp)}
            y={height - 12}
            textAnchor={index === 0 ? "start" : index === xTicks.length - 1 ? "end" : "middle"}
          >
            {formatDate(timestamp)}
          </text>
        ))}
        {groups.map((group) => {
          const color = stableCohortColor(group.cohort.key);
          const segments = metricSegments(group.points, metricKey, timestamps);
          return (
            <g key={group.cohort.key} style={{ color }}>
              {segments.map((segment, index) => (
                <polyline
                  key={`${group.cohort.key}-${index}`}
                  points={segment
                    .map((point) => `${xFor(point.timestamp ?? "")},${yFor(point.metrics[metricKey] as number)}`)
                    .join(" ")}
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.4"
                />
              ))}
              {segments.flat().map((point) => {
                const value = point.metrics[metricKey] as number;
                const chartPoint = { group, point, value, x: xFor(point.timestamp ?? ""), y: yFor(value) };
                const pointLabel = `${group.cohort.title}; ${formatMetricValue(metricKey, value, true)}; ${
                  group.cohort.hardware_label
                }; ${group.cohort.software_label}; recipe ${group.cohort.raw_ids.recipe_fingerprint}; commit ${shortSha(
                  point.commit_sha
                )}; ${runSourceLabel(point.run_source)}; ${formatTime(point.timestamp)}`;
                return (
                  <g
                    key={`${group.cohort.key}-${point.timestamp ?? ""}`}
                    onMouseEnter={() => setActivePoint(chartPoint)}
                    onMouseLeave={() => setActivePoint(null)}
                  >
                    <title>{pointLabel}</title>
                    <circle
                      className="point-hit-area"
                      cx={chartPoint.x}
                      cy={chartPoint.y}
                      r="12"
                      tabIndex={0}
                      aria-label={pointLabel}
                      onFocus={() => setActivePoint(chartPoint)}
                      onBlur={() => setActivePoint(null)}
                    />
                    <circle
                      className="compare-point-marker"
                      cx={chartPoint.x}
                      cy={chartPoint.y}
                      r="4"
                      fill="currentColor"
                    />
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      {unavailableGroups.length ? (
        <div className="metric-unavailable">
          Unavailable for {unavailableGroups.map((group) => group.cohort.title).join(", ")}.
        </div>
      ) : null}
      {activePoint ? (
        <div className="compare-point-details" aria-live="polite">
          <strong style={{ color: stableCohortColor(activePoint.group.cohort.key) }}>
            {activePoint.group.cohort.title} · {formatMetricValue(metricKey, activePoint.value, true)}
          </strong>
          <span>{activePoint.group.cohort.hardware_label}</span>
          <span>{activePoint.group.cohort.software_label}</span>
          <code>{cohortIdentifiers(activePoint.group.cohort)}</code>
          <span>Commit {shortSha(activePoint.point.commit_sha)}</span>
          <span>{runSourceLabel(activePoint.point.run_source)} · {formatTime(activePoint.point.timestamp)}</span>
        </div>
      ) : (
        <div className="compare-point-hint">Hover or focus a point for exact environment and run details.</div>
      )}
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
  const [hardwareFilter, setHardwareFilter] = useState(initialUrlState.hardware);
  const [softwareFilter, setSoftwareFilter] = useState(initialUrlState.software);
  const [recipeFilter, setRecipeFilter] = useState(initialUrlState.recipe);
  const [compareMode, setCompareMode] = useState(initialUrlState.compareMode);
  const [compareCohortKeys, setCompareCohortKeys] = useState(initialUrlState.compareCohorts);
  const [catalog, setCatalog] = useState<CohortCatalogResponse | null>(null);
  const [catalogResolved, setCatalogResolved] = useState(false);
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [trends, setTrends] = useState<TrendGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [overviewSortKey, setOverviewSortKey] = useState<OverviewSortKey>("status");
  const [overviewSortDirection, setOverviewSortDirection] = useState<SortDirection>("asc");

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
    fetchCohorts({
      modelId: modelFilter || undefined,
      gpuKey: gpuFilter || undefined,
      cohortKey: !compareMode && cohortFilter && cohortFilter !== ALL_COHORTS ? cohortFilter : undefined,
      runSource: sourceFilter || undefined,
      hardwareProfileId: hardwareFilter || undefined,
      softwareProfileId: softwareFilter || undefined,
      recipeFingerprint: recipeFilter || undefined
    })
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
        const nextHardware = resolveAdvancedFilterSelection(
          hardwareFilter,
          data.advanced_filters.hardware_profiles.map((option) => option.value)
        );
        const nextSoftware = resolveAdvancedFilterSelection(
          softwareFilter,
          data.advanced_filters.software_profiles.map((option) => option.value)
        );
        const nextRecipe = resolveAdvancedFilterSelection(
          recipeFilter,
          data.advanced_filters.recipes.map((option) => option.value)
        );
        if (nextHardware !== hardwareFilter || nextSoftware !== softwareFilter || nextRecipe !== recipeFilter) {
          setHardwareFilter(nextHardware);
          setSoftwareFilter(nextSoftware);
          setRecipeFilter(nextRecipe);
          return;
        }
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
  }, [
    modelFilter,
    gpuFilter,
    cohortFilter,
    sourceFilter,
    hardwareFilter,
    softwareFilter,
    recipeFilter,
    compareMode,
    refreshVersion
  ]);

  useEffect(() => {
    if (!catalogResolved || cohortFilter === null) {
      return;
    }
    let cancelled = false;
    const cohortKey = compareMode || cohortFilter === ALL_COHORTS ? undefined : cohortFilter;
    const query = {
      days,
      modelId: modelFilter || undefined,
      gpuKey: gpuFilter || undefined,
      cohortKey,
      runSource: sourceFilter || undefined,
      hardwareProfileId: hardwareFilter || undefined,
      softwareProfileId: softwareFilter || undefined,
      recipeFingerprint: recipeFilter || undefined
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
  }, [
    catalogResolved,
    cohortFilter,
    days,
    modelFilter,
    gpuFilter,
    sourceFilter,
    hardwareFilter,
    softwareFilter,
    recipeFilter,
    compareMode,
    refreshVersion
  ]);

  useEffect(() => {
    if (!catalogResolved || cohortFilter === null) {
      return;
    }
    const next = writeDashboardUrl(new URL(window.location.href), {
      days,
      model: modelFilter,
      gpu: gpuFilter,
      cohort: cohortFilter,
      source: sourceFilter,
      hardware: hardwareFilter,
      software: softwareFilter,
      recipe: recipeFilter,
      compareMode,
      compareCohorts: compareCohortKeys
    });
    window.history.replaceState(null, "", `${next.pathname}${next.search}${next.hash}`);
  }, [
    catalogResolved,
    cohortFilter,
    days,
    modelFilter,
    gpuFilter,
    sourceFilter,
    hardwareFilter,
    softwareFilter,
    recipeFilter,
    compareMode,
    compareCohortKeys
  ]);

  const latestRows = summary?.rows ?? [];
  const totalRuns = trends.reduce((total, group) => total + group.points.length, 0);
  const sync = summary?.sync;
  const selectedCohort = catalog?.cohorts.find((cohort) => cohort.key === cohortFilter);
  const allCohortsSelected = cohortFilter === ALL_COHORTS;
  const overviewRows = useMemo(
    () => sortCohortOverview(latestRows, overviewSortKey, overviewSortDirection),
    [latestRows, overviewSortKey, overviewSortDirection]
  );
  const observationCohortKeys = useMemo(
    () => new Set(trends.map((group) => group.cohort.key)),
    [trends]
  );
  const selectedCompareCohorts = useMemo(() => {
    const cohortsByKey = new Map((catalog?.cohorts ?? []).map((cohort) => [cohort.key, cohort]));
    return compareCohortKeys
      .map((key) => cohortsByKey.get(key))
      .filter((cohort): cohort is CohortCatalogItem => Boolean(cohort));
  }, [catalog, compareCohortKeys]);
  const selectedCompareGroups = useMemo(() => {
    const groupsByKey = new Map(trends.map((group) => [group.cohort.key, group]));
    return compareCohortKeys
      .map((key) => groupsByKey.get(key))
      .filter((group): group is TrendGroup => Boolean(group));
  }, [compareCohortKeys, trends]);

  useEffect(() => {
    if (!compareMode || !catalogResolved || loading || !catalog) {
      return;
    }
    setCompareCohortKeys((current) =>
      resolveCompatibleCompareSelection(current, catalog.cohorts, observationCohortKeys)
    );
  }, [compareMode, catalogResolved, loading, catalog, observationCohortKeys]);

  function changeOverviewSort(key: OverviewSortKey) {
    if (key === overviewSortKey) {
      setOverviewSortDirection((direction) => (direction === "asc" ? "desc" : "asc"));
      return;
    }
    setOverviewSortKey(key);
    setOverviewSortDirection(key === "status" ? "asc" : "desc");
  }

  function selectOverviewCohort(row: SummaryRow) {
    setCohortFilter(row.cohort.key);
  }

  function enterCompareMode(seedKey?: string) {
    const defaultSeed = cohortFilter && cohortFilter !== ALL_COHORTS ? cohortFilter : undefined;
    const seed = seedKey ?? defaultSeed;
    setCompareCohortKeys(seed ? [seed] : []);
    setCompareMode(true);
  }

  function toggleCompareCohort(cohort: CohortCatalogItem) {
    setCompareCohortKeys((current) => {
      if (current.includes(cohort.key)) {
        return current.filter((key) => key !== cohort.key);
      }
      if (current.length >= MAX_COMPARE_COHORTS) {
        return current;
      }
      return [...current, cohort.key];
    });
  }

  return (
    <main className="dashboard">
      <header className="topbar">
        <div>
          <p className="eyebrow">FastVideo CI</p>
          <h1>Performance Dashboard</h1>
        </div>
        <div className="topbar-actions">
          {!compareMode ? (
            <button className="compare-button" type="button" onClick={() => enterCompareMode()}>
              Compare cohorts
            </button>
          ) : null}
          <button className="refresh-button" onClick={refresh} disabled={refreshing || loading}>
            {refreshing ? "Refreshing" : "Refresh"}
          </button>
        </div>
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
            disabled={!catalogResolved || compareMode}
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

      <details className="advanced-filters">
        <summary>
          Advanced filters
          {hardwareFilter || softwareFilter || recipeFilter ? <span className="active-filter-count">Active</span> : null}
        </summary>
        <div className="advanced-filter-grid">
          <label>
            Hardware profile
            <select value={hardwareFilter} onChange={(event) => setHardwareFilter(event.target.value)}>
              <option value="">All hardware profiles</option>
              {(catalog?.advanced_filters.hardware_profiles ?? []).map((option) => (
                <option key={option.value} value={option.value}>
                  {advancedOptionLabel(option)}
                </option>
              ))}
            </select>
          </label>
          <label>
            Software profile
            <select value={softwareFilter} onChange={(event) => setSoftwareFilter(event.target.value)}>
              <option value="">All software profiles</option>
              {(catalog?.advanced_filters.software_profiles ?? []).map((option) => (
                <option key={option.value} value={option.value}>
                  {advancedOptionLabel(option)}
                </option>
              ))}
            </select>
          </label>
          <label>
            Recipe
            <select value={recipeFilter} onChange={(event) => setRecipeFilter(event.target.value)}>
              <option value="">All recipes</option>
              {(catalog?.advanced_filters.recipes ?? []).map((option) => (
                <option key={option.value} value={option.value}>
                  {advancedOptionLabel(option)}
                </option>
              ))}
            </select>
          </label>
          <button
            className="clear-advanced-button"
            type="button"
            disabled={!hardwareFilter && !softwareFilter && !recipeFilter}
            onClick={() => {
              setHardwareFilter("");
              setSoftwareFilter("");
              setRecipeFilter("");
            }}
          >
            Clear advanced filters
          </button>
        </div>
      </details>

      {selectedCohort && !compareMode ? (
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

      {compareMode ? (
        <section className="panel compare-panel">
          <div className="panel-header compare-header">
            <div>
              <h2>Compare Cohorts</h2>
              <span>Select two or three compatible exact cohorts.</span>
            </div>
            <button
              className="secondary-button"
              type="button"
              onClick={() => setCompareMode(false)}
            >
              Exit compare mode
            </button>
          </div>
          <div className="compare-picker" aria-label="Comparison cohorts">
            <div className="compare-picker-heading">
              <strong>{compareCohortKeys.length} of {MAX_COMPARE_COHORTS} selected</strong>
              <span>Availability reflects Model, GPU, Source, date, and advanced filters.</span>
            </div>
            {(catalog?.cohorts ?? []).length === 0 ? (
              <div className="empty">No cohorts match the active filters.</div>
            ) : (
              <div className="compare-options">
                {(catalog?.cohorts ?? []).map((cohort) => {
                  const selected = compareCohortKeys.includes(cohort.key);
                  const reason = selected
                    ? null
                    : comparisonUnavailableReason(cohort, selectedCompareCohorts, observationCohortKeys);
                  const atLimit = !selected && compareCohortKeys.length >= MAX_COMPARE_COHORTS;
                  return (
                    <label className={`compare-option ${reason || atLimit ? "compare-option-disabled" : ""}`} key={cohort.key}>
                      <input
                        type="checkbox"
                        checked={selected}
                        disabled={Boolean(reason) || atLimit}
                        onChange={() => toggleCompareCohort(cohort)}
                      />
                      <span className="legend-swatch" style={{ background: stableCohortColor(cohort.key) }} />
                      <span className="compare-option-label">
                        <strong>{cohort.title} · {cohort.gpu_label}</strong>
                        <span>{cohort.hardware_label}</span>
                        <span>{cohort.software_label}</span>
                        <code>{cohortIdentifiers(cohort)}</code>
                        {reason ? <em>{reason}</em> : null}
                        {atLimit ? <em>Remove a selected cohort before adding another.</em> : null}
                      </span>
                    </label>
                  );
                })}
              </div>
            )}
          </div>
          {selectedCompareGroups.length < 2 ? (
            <div className="compare-prompt" role="status">
              Select at least two compatible cohorts to render a comparison. Up to three cohorts are supported.
            </div>
          ) : (
            <div className="compare-results">
              <div className="compare-legend" aria-label="Cohort legend">
                {selectedCompareCohorts.map((cohort) => (
                  <div key={cohort.key}>
                    <span className="legend-swatch" style={{ background: stableCohortColor(cohort.key) }} />
                    <span>
                      <strong>{cohort.title}</strong>
                      <small>{cohort.gpu_label} · {cohort.software_label}</small>
                    </span>
                  </div>
                ))}
              </div>
              <div className="compare-chart-grid">
                {METRIC_KEYS.map((metricKey) => (
                  <article className="compare-chart-card" key={metricKey}>
                    <h3>{metricLabel(metricKey)}</h3>
                    <CompareChart groups={selectedCompareGroups} metricKey={metricKey} />
                  </article>
                ))}
              </div>
            </div>
          )}
        </section>
      ) : allCohortsSelected ? (
        <section className="panel cohort-overview-panel">
          <div className="panel-header">
            <h2>Cohort Overview</h2>
            <div className="panel-header-actions">
              <span>{latestRows.length} exact comparison cohorts</span>
              <button className="secondary-button" type="button" onClick={() => enterCompareMode()}>
                Compare cohorts
              </button>
            </div>
          </div>
          {latestRows.length === 0 ? (
            <div className="empty">No cohorts match the selected filters.</div>
          ) : (
            <>
              {latestRows.length === 1 ? (
                <div className="single-cohort-note">One exact cohort matches. Select its row to inspect detailed trends.</div>
              ) : null}
              <div className="table-wrap">
                <table className="cohort-overview-table">
                  <thead>
                    <tr>
                      <th aria-sort={overviewSortKey === "status" ? (overviewSortDirection === "asc" ? "ascending" : "descending") : "none"}>
                        <SortHeader
                          label="Status"
                          sortKey="status"
                          activeKey={overviewSortKey}
                          direction={overviewSortDirection}
                          onChange={changeOverviewSort}
                        />
                      </th>
                      <th>Model</th>
                      <th>GPU configuration</th>
                      <th>Benchmark cohort</th>
                      <th aria-sort={overviewSortKey === "latest" ? (overviewSortDirection === "asc" ? "ascending" : "descending") : "none"}>
                        <SortHeader
                          label="Latest run"
                          sortKey="latest"
                          activeKey={overviewSortKey}
                          direction={overviewSortDirection}
                          onChange={changeOverviewSort}
                        />
                      </th>
                      <th>Source / schedule</th>
                      <th aria-sort={overviewSortKey === "latency" ? (overviewSortDirection === "asc" ? "ascending" : "descending") : "none"}>
                        <SortHeader
                          label="Latency"
                          sortKey="latency"
                          activeKey={overviewSortKey}
                          direction={overviewSortDirection}
                          onChange={changeOverviewSort}
                        />
                      </th>
                      <th aria-sort={overviewSortKey === "throughput" ? (overviewSortDirection === "asc" ? "ascending" : "descending") : "none"}>
                        <SortHeader
                          label="Throughput"
                          sortKey="throughput"
                          activeKey={overviewSortKey}
                          direction={overviewSortDirection}
                          onChange={changeOverviewSort}
                        />
                      </th>
                      <th aria-sort={overviewSortKey === "memory" ? (overviewSortDirection === "asc" ? "ascending" : "descending") : "none"}>
                        <SortHeader
                          label="Memory"
                          sortKey="memory"
                          activeKey={overviewSortKey}
                          direction={overviewSortDirection}
                          onChange={changeOverviewSort}
                        />
                      </th>
                      <th>Baseline</th>
                    </tr>
                  </thead>
                  <tbody>
                    {overviewRows.map((row) => (
                      <tr
                        className="selectable-row"
                        key={row.cohort.key}
                        tabIndex={0}
                        onClick={() => selectOverviewCohort(row)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            selectOverviewCohort(row);
                          }
                        }}
                        aria-label={`Inspect ${row.model_id}, ${row.cohort.title}, ${row.cohort.gpu_label}`}
                      >
                        <td>
                          <span className={`badge ${row.status}`}>{row.status}</span>
                        </td>
                        <td>{row.model_id}</td>
                        <td>{row.cohort.gpu_label}</td>
                        <td><CohortLabel cohort={row.cohort} /></td>
                        <td>{formatTime(row.timestamp)}</td>
                        <td>
                          <span className={`source-badge source-${row.run_source}`}>
                            {runSourceLabel(row.run_source)}
                          </span>
                          <span className="schedule-label">{row.test_scope || "Unavailable"}</span>
                        </td>
                        <td>{formatOverviewMetric("latency", row.metrics.latency?.current)}</td>
                        <td>{formatOverviewMetric("throughput", row.metrics.throughput?.current)}</td>
                        <td>{formatOverviewMetric("memory", row.metrics.memory?.current)}</td>
                        <td>{row.baseline_eligible ? "Eligible" : "Excluded"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>
      ) : (
        <>
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
                          <span className={`source-badge source-${row.run_source}`}>
                            {runSourceLabel(row.run_source)}
                          </span>
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
        </>
      )}
    </main>
  );
}
