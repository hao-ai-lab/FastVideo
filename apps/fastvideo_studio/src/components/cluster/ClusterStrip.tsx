'use client';

import type { ClusterNode } from '@/lib/api';
import { cn } from '@/lib/utils';

export function formatGib(mib: number): string {
  return `${(mib / 1024).toFixed(1)} GiB`;
}

/** Bar fill class by load: blue when idle, amber under pressure, rose hot. */
export function utilizationColor(percent: number): string {
  if (percent >= 85) return 'bg-rose-500';
  if (percent >= 50) return 'bg-amber-500';
  return 'bg-accent-blue';
}

export function clampPercent(percent: number): number {
  return Math.max(0, Math.min(100, percent));
}

function GpuSegment({
  index,
  utilization,
  memUsedMib,
  memTotalMib,
}: {
  index: number;
  utilization: number;
  memUsedMib: number;
  memTotalMib: number;
}) {
  const memPercent =
    memTotalMib > 0 ? clampPercent((memUsedMib / memTotalMib) * 100) : 0;
  const label =
    `GPU ${index}: ${utilization}% utilization, ` +
    `${formatGib(memUsedMib)} / ${formatGib(memTotalMib)} VRAM`;
  return (
    <div
      role="img"
      aria-label={label}
      title={label}
      className="flex w-10 shrink-0 flex-col gap-0.5"
    >
      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div
          className={cn('h-full rounded-full', utilizationColor(utilization))}
          style={{ width: `${clampPercent(utilization)}%` }}
        />
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div
          className={cn(
            'h-full rounded-full',
            memPercent >= 90 ? 'bg-rose-500' : 'bg-accent-blue',
          )}
          style={{ width: `${memPercent}%` }}
        />
      </div>
    </div>
  );
}

/** One compact line per node: hostname + tiny util/VRAM bars per GPU. */
export default function ClusterStrip({ nodes }: { nodes: ClusterNode[] }) {
  return (
    <div className="flex flex-col gap-2">
      {nodes.map((node, i) => (
        <div
          key={`${node.hostname}-${i}`}
          className="flex items-center gap-3"
        >
          <span className="w-40 shrink-0 truncate text-xs font-medium">
            {node.hostname}
          </span>
          <div className="flex min-w-0 flex-wrap items-center gap-1.5">
            {node.gpus.map((gpu) => (
              <GpuSegment
                key={gpu.index}
                index={gpu.index}
                utilization={gpu.utilization}
                memUsedMib={gpu.memory_used_mib}
                memTotalMib={gpu.memory_total_mib}
              />
            ))}
            {node.gpus.length === 0 && (
              <span className="text-xs text-muted-foreground">no GPUs</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
