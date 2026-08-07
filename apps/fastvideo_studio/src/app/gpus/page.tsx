'use client';

import * as React from 'react';
import { AlertTriangle, Info } from 'lucide-react';

import ClusterStrip, {
  clampPercent,
  formatGib,
  utilizationColor,
} from '@/components/cluster/ClusterStrip';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  getClusterStatus,
  type ClusterGpu,
  type ClusterNode,
  type ClusterSnapshot,
} from '@/lib/api';
import { cn } from '@/lib/utils';

const POLL_INTERVAL_MS = 5000;

function Meter({
  label,
  percent,
  detail,
  fillClass,
}: {
  label: string;
  percent: number;
  detail: string;
  fillClass: string;
}) {
  const clamped = clampPercent(percent);
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between gap-2 text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium tabular-nums text-foreground">
          {detail}
        </span>
      </div>
      <div
        role="meter"
        aria-label={label}
        aria-valuenow={Math.round(clamped)}
        aria-valuemin={0}
        aria-valuemax={100}
        className="h-1.5 overflow-hidden rounded-full bg-muted"
      >
        <div
          className={cn(
            'h-full rounded-full transition-[width] duration-500',
            fillClass,
          )}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}

function GpuRow({ gpu }: { gpu: ClusterGpu }) {
  const memPercent =
    gpu.memory_total_mib > 0
      ? (gpu.memory_used_mib / gpu.memory_total_mib) * 100
      : 0;
  return (
    <div className="grid items-center gap-x-6 gap-y-2 border-t border-border pt-3 first:border-t-0 first:pt-0 md:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)_minmax(0,1.6fr)_auto]">
      <div className="flex min-w-0 items-baseline gap-2">
        <span className="min-w-0 truncate text-sm font-semibold">
          {gpu.name}
        </span>
        <span className="shrink-0 text-xs font-medium uppercase tracking-wider text-muted-foreground">
          GPU {gpu.index}
        </span>
      </div>
      <Meter
        label="Utilization"
        percent={gpu.utilization}
        detail={`${gpu.utilization}%`}
        fillClass={utilizationColor(gpu.utilization)}
      />
      <Meter
        label="VRAM"
        percent={memPercent}
        detail={`${gpu.memory_used_mib} / ${gpu.memory_total_mib} MiB (${formatGib(gpu.memory_used_mib)} / ${formatGib(gpu.memory_total_mib)})`}
        fillClass={memPercent >= 90 ? 'bg-rose-500' : 'bg-accent-blue'}
      />
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs tabular-nums text-muted-foreground md:w-28 md:justify-end">
        {gpu.temperature_c != null && <span>{gpu.temperature_c}°C</span>}
        {gpu.power_watts != null && (
          <span>
            {Math.round(gpu.power_watts)} W
            {gpu.power_limit_watts != null &&
              ` / ${Math.round(gpu.power_limit_watts)} W`}
          </span>
        )}
      </div>
    </div>
  );
}

function NodeSection({ node }: { node: ClusterNode }) {
  return (
    <Card>
      <CardContent className="flex flex-col gap-3 p-5">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="min-w-0 truncate text-sm font-semibold">
            {node.hostname}
          </span>
          {node.ip && (
            <span className="text-xs tabular-nums text-muted-foreground">
              {node.ip}
            </span>
          )}
          {node.is_this_host && <Badge variant="secondary">API host</Badge>}
          <span className="ml-auto text-xs tabular-nums text-muted-foreground">
            {node.cpus != null && `${Math.round(node.cpus)} CPUs`}
            {node.cpus != null && node.ray_gpus != null && ' · '}
            {node.ray_gpus != null && `${Math.round(node.ray_gpus)} ray GPUs`}
          </span>
        </div>
        {!node.available && (
          <p className="text-sm text-muted-foreground">
            GPU telemetry unavailable
            {node.error ? `: ${node.error}` : '.'}
          </p>
        )}
        {node.gpus.map((gpu) => (
          <GpuRow key={gpu.index} gpu={gpu} />
        ))}
      </CardContent>
    </Card>
  );
}

export default function GpusPage() {
  const [snapshot, setSnapshot] = React.useState<ClusterSnapshot | null>(null);
  const [fetchError, setFetchError] = React.useState<string | null>(null);
  const [retryToken, setRetryToken] = React.useState(0);

  React.useEffect(() => {
    let mounted = true;
    let inFlight = false;

    async function poll() {
      if (inFlight || document.hidden) return;
      inFlight = true;
      try {
        const next = await getClusterStatus();
        if (mounted) {
          setSnapshot(next);
          setFetchError(null);
        }
      } catch {
        if (mounted) {
          setFetchError(
            'Cluster status could not be refreshed. The values below may be stale.',
          );
        }
      } finally {
        inFlight = false;
      }
    }

    poll();
    const interval = setInterval(poll, POLL_INTERVAL_MS);
    // Refresh immediately when the tab becomes visible again (polls are
    // skipped while hidden).
    document.addEventListener('visibilitychange', poll);
    return () => {
      mounted = false;
      clearInterval(interval);
      document.removeEventListener('visibilitychange', poll);
    };
  }, [retryToken]);

  let body: React.ReactNode;
  if (fetchError && !snapshot) {
    body = (
      <div
        role="alert"
        className="flex flex-col items-center gap-3 py-8 text-center"
      >
        <AlertTriangle className="size-6 text-destructive" aria-hidden />
        <p className="text-muted-foreground">
          Could not reach the API server. Cluster status needs the Studio API
          server running.
        </p>
        <Button
          type="button"
          variant="outline"
          onClick={() => setRetryToken((token) => token + 1)}
        >
          Try Again
        </Button>
      </div>
    );
  } else if (!snapshot) {
    body = <p className="py-8 text-center text-muted-foreground">Loading…</p>;
  } else {
    body = (
      <div className="flex flex-col gap-4">
        <header className="flex flex-wrap items-center gap-3">
          <h1 className="text-lg font-semibold">Cluster</h1>
          <Badge variant="outline">
            {snapshot.mode === 'ray' ? 'ray cluster' : 'local host only'}
          </Badge>
          {snapshot.resources && (
            <span className="text-sm tabular-nums text-muted-foreground">
              {Math.round(snapshot.resources.gpus_available)} /{' '}
              {Math.round(snapshot.resources.gpus_total)} GPUs available
            </span>
          )}
        </header>
        {snapshot.error && (
          <div
            role="status"
            className="flex flex-wrap items-center gap-3 rounded-lg border border-blue-400/40 bg-blue-500/10 px-3 py-2 text-sm"
          >
            <Info className="size-4 shrink-0 text-blue-600" aria-hidden />
            <span className="min-w-0 flex-1">{snapshot.error}</span>
          </div>
        )}
        {fetchError && (
          <div
            role="status"
            aria-live="polite"
            className="flex flex-wrap items-center gap-3 rounded-lg border border-amber-500/50 bg-amber-500/10 px-3 py-2 text-sm"
          >
            <AlertTriangle className="size-4 text-amber-600" aria-hidden />
            <span className="min-w-0 flex-1">{fetchError}</span>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setRetryToken((token) => token + 1)}
            >
              Refresh Now
            </Button>
          </div>
        )}
        <ClusterStrip nodes={snapshot.nodes} />
        {snapshot.nodes.map((node, i) => (
          <NodeSection key={`${node.hostname}-${i}`} node={node} />
        ))}
      </div>
    );
  }

  return (
    <div className="mx-auto flex w-full max-w-[1100px] flex-col gap-6 px-4 pb-12 pt-6">
      {body}
    </div>
  );
}
