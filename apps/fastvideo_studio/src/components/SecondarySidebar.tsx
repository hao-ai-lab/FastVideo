'use client';

import * as React from 'react';
import { X } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { useResizable } from '@/hooks/useResizable';
import { downloadJobLog, getJobLogs } from '@/lib/api';
import type { Job } from '@/lib/types';
import { cn } from '@/lib/utils';

const SIDEBAR_MIN_WIDTH = 280;
const SIDEBAR_MAX_WIDTH = 750;
const POLL_INTERVAL_MS = 2000;

export default function SecondarySidebar({
  job,
  onClose,
  onWidthChange,
}: {
  job: Job;
  onClose: () => void;
  onWidthChange?: (w: number) => void;
}) {
  const [width, setWidth] = React.useState(360);
  const [isDragging, setIsDragging] = React.useState(false);
  const [isLoading, setIsLoading] = React.useState(false);
  const [logs, setLogs] = React.useState<string[]>([]);

  // Race-guard refs (mirror the Svelte original): the cursor + accumulated
  // lines live here so in-flight polls don't read stale React state, and
  // pollingLock prevents overlapping fetches. Do NOT swap these for effect
  // deps — they must update synchronously, outside React's render cycle.
  const stateRef = React.useRef<{ logs: string[]; logAfter: number }>({
    logs: [],
    logAfter: 0,
  });
  const pollingLock = React.useRef(false);
  const previousJobId = React.useRef<string | null>(null);
  const previousStatus = React.useRef<string | null>(null);
  const consoleRef = React.useRef<HTMLPreElement | null>(null);

  const { onMouseDown } = useResizable({
    // Right-docked panel with the drag handle on its left edge: dragging the
    // handle left must grow the panel, which is `edge: 'right'` (matches the
    // Svelte original). `edge: 'left'` would invert the drag.
    edge: 'right',
    minWidth: SIDEBAR_MIN_WIDTH,
    maxWidth: SIDEBAR_MAX_WIDTH,
    getWidth: () => width,
    onWidth: setWidth,
    onDragChange: setIsDragging,
  });

  React.useEffect(() => {
    onWidthChange?.(width);
  }, [width, onWidthChange]);

  // Auto-scroll the console to the bottom whenever new lines land. Runs after
  // commit so scrollHeight reflects the freshly-rendered output.
  React.useEffect(() => {
    const el = consoleRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logs]);

  // Reset on job switch or restart. Declared before the polling effect so the
  // cursor is cleared before the next poll reads it.
  React.useEffect(() => {
    const wasTerminal =
      previousStatus.current === 'failed' ||
      previousStatus.current === 'stopped' ||
      previousStatus.current === 'completed';
    const isRestarting =
      previousJobId.current === job.id &&
      wasTerminal &&
      (job.status === 'pending' || job.status === 'running');

    if (previousJobId.current !== job.id || isRestarting) {
      stateRef.current.logs = [];
      stateRef.current.logAfter = 0;
      setLogs([]);
    }
    previousJobId.current = job.id;
    previousStatus.current = job.status;
  }, [job.id, job.status]);

  React.useEffect(() => {
    const shouldPoll = job.status === 'running' || job.status === 'pending';
    let pollInterval: ReturnType<typeof setInterval> | null = null;
    let mounted = true;

    async function pollLogs() {
      if (!mounted || pollingLock.current) return;
      pollingLock.current = true;
      try {
        const logData = await getJobLogs(job.id, stateRef.current.logAfter);
        if (mounted && logData.lines.length > 0) {
          stateRef.current.logs = [...stateRef.current.logs, ...logData.lines];
          stateRef.current.logAfter = logData.total;
          setLogs(stateRef.current.logs);
        }
      } catch (e) {
        console.error('Failed to fetch logs:', e);
      } finally {
        pollingLock.current = false;
      }
    }

    pollLogs();
    if (shouldPoll) pollInterval = setInterval(pollLogs, POLL_INTERVAL_MS);

    return () => {
      mounted = false;
      if (pollInterval) clearInterval(pollInterval);
    };
  }, [job.id, job.status]);

  async function handleDownloadLog() {
    if (isLoading) return;
    setIsLoading(true);
    try {
      const blob = await downloadJobLog(job.id);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `job_${job.id}.log`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Failed to download log:', err);
      alert(err instanceof Error ? err.message : 'Failed to download log');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <aside
      className="fixed bottom-0 right-0 top-[var(--header-height)] z-50 flex max-h-[calc(100vh-var(--header-height))] min-w-[280px] shrink-0 flex-col border-l border-border bg-card"
      style={{ width, maxWidth: SIDEBAR_MAX_WIDTH }}
    >
      <div className="flex items-center justify-between border-b border-border px-5 py-4">
        <h2 className="m-0 text-base font-semibold text-foreground">
          Job Details
        </h2>
        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={handleDownloadLog}
            disabled={isLoading || !job.log_file_path}
            title="Download log file"
          >
            Download Log
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            onClick={onClose}
            title="Close"
            aria-label="Close"
          >
            <X className="h-[18px] w-[18px]" />
          </Button>
        </div>
      </div>

      <div className="flex min-h-0 flex-1 flex-col px-5 py-4">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Console Output
          </span>
          {job.status === 'running' && (
            <span className="text-[0.7rem] font-medium text-[var(--green)]">
              ● Live
            </span>
          )}
        </div>
        <pre
          ref={consoleRef}
          className="m-0 min-h-0 flex-1 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-background p-3 font-mono text-xs leading-normal text-foreground"
        >
          {logs.length === 0 ? (
            <span className="italic text-muted-foreground">
              {job.status === 'running'
                ? 'Waiting for logs...'
                : 'No logs available'}
            </span>
          ) : (
            logs.join('\n')
          )}
        </pre>
      </div>

      <div
        role="presentation"
        onMouseDown={onMouseDown}
        className={cn(
          'absolute bottom-0 left-0 top-0 z-[1] w-1.5 cursor-col-resize hover:bg-accent/20',
          isDragging && 'bg-accent/20',
        )}
      />
    </aside>
  );
}
