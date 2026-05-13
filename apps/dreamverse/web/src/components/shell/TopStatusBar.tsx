'use client';

import React, { useMemo } from 'react';

import { Badge } from '@/components/ui/badge';

interface Props {
  modeLabel?: string;
  title?: string;
  connected?: boolean;
  connecting?: boolean;
  sessionStarted?: boolean;
  gpuAssigned?: boolean;
  queuePosition?: number;
  livePromptRewriteMode?: boolean;
  rewritingSeedPrompts?: boolean;
  timeLeft?: number | null;
  ttffStartAtMs?: number | null;
  ttffValueMs?: number | null;
  videoGapMs?: number | null;
  formatTime?: (seconds: number) => string;
  formatDurationMs?: (durationMs: number) => string;
}

export default function TopStatusBar({
  modeLabel = 'Standard Mode',
  title = 'Preset-driven video continuation',
  connected = false,
  connecting = false,
  sessionStarted = false,
  gpuAssigned = false,
  queuePosition = 0,
  livePromptRewriteMode = false,
  rewritingSeedPrompts = false,
  timeLeft = null,
  ttffStartAtMs = null,
  ttffValueMs = null,
  videoGapMs = null,
  formatTime = (seconds) => `${seconds}`,
  formatDurationMs = (durationMs) => `${durationMs}`,
}: Props) {
  const connectionLabel = connected
    ? 'Connected'
    : connecting
      ? 'Connecting'
      : 'Offline';

  const sessionLabel = useMemo(() => {
    if (!sessionStarted) return 'Idle';
    if (connecting) return 'Connecting';
    if (sessionStarted && queuePosition > 0 && !gpuAssigned)
      return `Queue ${queuePosition}`;
    if (connected && !gpuAssigned) return 'Loading model';
    return 'Session active';
  }, [sessionStarted, connecting, queuePosition, gpuAssigned, connected]);

  const gpuLabel = gpuAssigned
    ? 'GPU ready'
    : sessionStarted
      ? 'Awaiting GPU'
      : 'GPU idle';

  const rewriteLabel = livePromptRewriteMode
    ? rewritingSeedPrompts
      ? 'Rewriting window'
      : 'Rewrite mode'
    : null;

  const connectionVariant = connected
    ? 'success'
    : connecting
      ? 'warning'
      : 'outline';

  const gpuVariant = gpuAssigned
    ? 'success'
    : sessionStarted
      ? 'warning'
      : 'secondary';

  const timeVariant =
    timeLeft !== null && timeLeft <= 30 ? 'warning' : 'secondary';

  return (
    <header className="rounded-2xl border border-border bg-card px-5 py-4 shadow-sm backdrop-blur-sm">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
          <div
            className="flex items-center rounded-2xl border border-border bg-secondary px-4 py-3"
            aria-label="FastVideo"
          >
            <img
              src="/logo.svg"
              alt="FastVideo"
              className="h-8 w-auto sm:h-9"
            />
          </div>

          <div className="space-y-1">
            <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-sky-200/70">
              {modeLabel}
            </p>
            <h1 className="text-lg font-semibold text-foreground sm:text-xl">
              {title}
            </h1>
          </div>
        </div>

        <div className="flex flex-col gap-3 xl:items-end">
          <div className="flex flex-wrap gap-2" aria-label="Session status">
            <Badge variant={connectionVariant}>{connectionLabel}</Badge>
            <Badge variant="secondary">{sessionLabel}</Badge>
            <Badge variant={gpuVariant}>{gpuLabel}</Badge>
            {rewriteLabel && <Badge variant="default">{rewriteLabel}</Badge>}
          </div>

          <div className="flex flex-wrap gap-2">
            {timeLeft !== null && (
              <Badge
                variant={timeVariant}
                className="rounded-xl px-3 py-1 text-xs font-medium normal-case tracking-normal"
              >
                Time left: {formatTime(timeLeft)}
              </Badge>
            )}

            {ttffStartAtMs !== null && ttffValueMs !== null && (
              <Badge
                variant="secondary"
                className="rounded-xl px-3 py-1 text-xs font-medium normal-case tracking-normal"
              >
                Time to first frame: {formatDurationMs(ttffValueMs)}
              </Badge>
            )}

            {videoGapMs !== null && (
              <Badge
                variant="outline"
                className="rounded-xl px-3 py-1 text-xs font-medium normal-case tracking-normal"
              >
                Time between videos: {formatDurationMs(videoGapMs)}
              </Badge>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
