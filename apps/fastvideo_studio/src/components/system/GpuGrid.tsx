'use client';

import * as React from 'react';

import { Card, CardContent } from '@/components/ui/card';
import { getGpus, type GpuInfo, type GpuSnapshot } from '@/lib/api';
import { cn } from '@/lib/utils';

const POLL_INTERVAL_MS = 3000;

function formatGib(mib: number): string {
  return `${(mib / 1024).toFixed(1)} GiB`;
}

function Meter({
  label,
  percent,
  detail,
  warnAt,
}: {
  label: string;
  percent: number;
  detail: string;
  /** Turn the fill rose at this percentage (e.g. VRAM pressure). */
  warnAt?: number;
}) {
  const clamped = Math.max(0, Math.min(100, percent));
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
            'h-full rounded-full bg-accent-blue transition-[width] duration-500',
            warnAt !== undefined && clamped >= warnAt && 'bg-rose-500',
          )}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}

function GpuCard({ gpu }: { gpu: GpuInfo }) {
  const memPercent =
    gpu.memory_total_mib > 0
      ? (gpu.memory_used_mib / gpu.memory_total_mib) * 100
      : 0;
  return (
    <Card>
      <CardContent className="flex flex-col gap-4 p-5">
        <div className="flex items-baseline justify-between gap-2">
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
        />
        <Meter
          label="Memory"
          percent={memPercent}
          warnAt={90}
          detail={`${formatGib(gpu.memory_used_mib)} / ${formatGib(gpu.memory_total_mib)}`}
        />
        <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs tabular-nums text-muted-foreground">
          {gpu.temperature_c != null && <span>{gpu.temperature_c}°C</span>}
          {gpu.power_watts != null && (
            <span>
              {Math.round(gpu.power_watts)} W
              {gpu.power_limit_watts != null &&
                ` / ${Math.round(gpu.power_limit_watts)} W`}
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default function GpuGrid() {
  const [snapshot, setSnapshot] = React.useState<GpuSnapshot | null>(null);
  const [fetchError, setFetchError] = React.useState(false);

  React.useEffect(() => {
    let mounted = true;
    let inFlight = false;

    async function poll() {
      if (inFlight) return;
      inFlight = true;
      try {
        const next = await getGpus();
        if (mounted) {
          setSnapshot(next);
          setFetchError(false);
        }
      } catch {
        if (mounted) setFetchError(true);
      } finally {
        inFlight = false;
      }
    }

    poll();
    const interval = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  if (fetchError && !snapshot) {
    return (
      <p className="py-8 text-center text-muted-foreground">
        Could not reach the API server. GPU status needs the studio API server
        running.
      </p>
    );
  }
  if (!snapshot) {
    return <p className="py-8 text-center text-muted-foreground">Loading…</p>;
  }
  if (!snapshot.available) {
    return (
      <p className="py-8 text-center text-muted-foreground">
        GPU telemetry unavailable
        {snapshot.error ? `: ${snapshot.error}` : '.'}
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {fetchError && (
        <p className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-600 dark:text-amber-400">
          Lost contact with the API server — showing the last known values.
        </p>
      )}
      <div className="grid gap-4 [grid-template-columns:repeat(auto-fill,minmax(280px,1fr))]">
        {snapshot.gpus.map((gpu) => (
          <GpuCard key={gpu.index} gpu={gpu} />
        ))}
      </div>
    </div>
  );
}
