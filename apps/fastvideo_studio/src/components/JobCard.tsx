'use client';

import * as React from 'react';

import { Button } from '@/components/ui/button';
import { useStore } from '@/hooks/useStore';
import {
  deleteJob,
  downloadJobVideo,
  startJob,
  stopJob,
} from '@/lib/api';
import type { Job } from '@/lib/types';
import { cn } from '@/lib/utils';
import { activeJobStore, setActiveJobId } from '@/stores/activeJob';

interface JobCardProps {
  job: Job;
  onJobUpdated?: () => void;
}

function formatDuration(seconds: number): string {
  const roundedSeconds = Math.round(seconds);
  if (roundedSeconds < 60) return `${roundedSeconds}s`;
  if (roundedSeconds < 3600) {
    const mins = Math.floor(roundedSeconds / 60);
    const secs = roundedSeconds % 60;
    return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
  }
  const hours = Math.floor(roundedSeconds / 3600);
  const mins = Math.floor((roundedSeconds % 3600) / 60);
  return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
}

function computeElapsed(job: Job, currentTime: number): string | null {
  if (!job.started_at) return null;
  const endTime =
    job.status === 'running' ? currentTime : (job.finished_at ?? 0);
  if (!endTime && job.status !== 'running') return null;
  const startedAtMs =
    job.started_at < 1e12 ? job.started_at * 1000 : job.started_at;
  const endTimeMs = endTime < 1e12 ? endTime * 1000 : endTime;
  const elapsedSeconds = (endTimeMs - startedAtMs) / 1000;
  if (elapsedSeconds <= 0) return null;
  return formatDuration(elapsedSeconds);
}

const BADGE_CLASSES: Record<string, string> = {
  pending: 'bg-secondary text-muted-foreground',
  running: 'bg-amber-500/20 text-amber-300',
  completed: 'bg-emerald-500/20 text-emerald-300',
  ready: 'bg-emerald-500/20 text-emerald-300',
  failed: 'bg-rose-500/20 text-rose-300',
  stopped: 'bg-secondary text-muted-foreground',
  preprocessing: 'bg-blue-500/20 text-blue-300',
};

export default function JobCard({ job, onJobUpdated }: JobCardProps) {
  const { activeJobId } = useStore(activeJobStore);
  const isSelected = activeJobId === job.id;

  const [isLoading, setIsLoading] = React.useState(false);
  const [currentTime, setCurrentTime] = React.useState(() => Date.now());

  const elapsedTime = computeElapsed(job, currentTime);

  React.useEffect(() => {
    if (job.status !== 'running' || !job.started_at) return;
    const interval = setInterval(() => setCurrentTime(Date.now()), 1000);
    return () => clearInterval(interval);
  }, [job.status, job.started_at]);

  async function handleStart(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (isLoading || job.status === 'running' || job.status === 'completed')
      return;
    setIsLoading(true);
    try {
      await startJob(job.id);
      onJobUpdated?.();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to start job');
    } finally {
      setIsLoading(false);
    }
  }

  async function handleStop(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (isLoading || job.status !== 'running') return;
    setIsLoading(true);
    try {
      await stopJob(job.id);
      onJobUpdated?.();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to stop job');
    } finally {
      setIsLoading(false);
    }
  }

  async function handleDelete(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (isLoading) return;
    if (!confirm('Delete this job?')) return;
    setIsLoading(true);
    try {
      await deleteJob(job.id);
      onJobUpdated?.();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete job');
    } finally {
      setIsLoading(false);
    }
  }

  function handleSelectJob(e: React.MouseEvent | React.KeyboardEvent) {
    if ((e.target as HTMLElement).closest('button')) return;
    setActiveJobId(isSelected ? null : job.id);
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleSelectJob(e);
    }
  }

  async function handleDownloadVideo(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (isLoading || !job.output_path) return;
    setIsLoading(true);
    try {
      const blob = await downloadJobVideo(job.id);
      const ext = job.output_path.endsWith('.png') ? 'png' : 'mp4';
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `job_${job.id}.${ext}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to download video');
    } finally {
      setIsLoading(false);
    }
  }

  const badgeClass = BADGE_CLASSES[job.status] ?? 'bg-secondary text-muted-foreground';

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={handleSelectJob}
      onKeyDown={handleKeyDown}
      className={cn(
        'mb-3 flex cursor-pointer flex-col gap-2.5 rounded-lg border bg-background p-4 transition-colors last:mb-0',
        isSelected
          ? 'border-primary bg-primary/5'
          : 'border-border hover:border-muted-foreground/40',
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-[0.95rem] font-semibold text-foreground">
          {job.model_id}
        </span>
        <span
          className={cn(
            'inline-flex items-center rounded-full px-2 py-0.5 text-[0.7rem] font-bold uppercase tracking-wide',
            badgeClass,
          )}
        >
          {job.status}
        </span>
      </div>
      <p className="max-w-full overflow-hidden text-ellipsis whitespace-nowrap text-sm text-muted-foreground">
        {job.prompt}
      </p>
      <div className="flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
        {job.job_type === 'inference' ? (
          <>
            <span>{job.num_frames} frames</span>
            <span>
              {job.height}×{job.width}
            </span>
          </>
        ) : (
          <span>{job.workload_type?.replace(/_/g, ' ') ?? job.job_type}</span>
        )}
        {elapsedTime && <span>⏱ {elapsedTime}</span>}
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        {job.status === 'running' ? (
          <Button
            size="sm"
            onClick={handleStop}
            disabled={isLoading}
            className="border-transparent bg-amber-500 text-black shadow-md hover:bg-amber-400"
          >
            Stop
          </Button>
        ) : job.status === 'failed' ? (
          <Button
            size="sm"
            onClick={handleStart}
            disabled={isLoading}
            className="border-transparent bg-emerald-600 text-white shadow-md hover:bg-emerald-500"
          >
            Restart
          </Button>
        ) : job.status === 'pending' || job.status === 'stopped' ? (
          <Button
            size="sm"
            onClick={handleStart}
            disabled={isLoading}
            className="border-transparent bg-emerald-600 text-white shadow-md hover:bg-emerald-500"
          >
            Start
          </Button>
        ) : null}
        {job.status === 'completed' &&
          job.output_path &&
          job.job_type === 'inference' && (
            <Button
              size="sm"
              variant="outline"
              onClick={handleDownloadVideo}
              disabled={isLoading}
              title="Download video"
            >
              Download Video
            </Button>
          )}
        <Button
          size="sm"
          variant="destructive"
          onClick={handleDelete}
          disabled={isLoading}
        >
          Delete
        </Button>
      </div>
    </div>
  );
}
