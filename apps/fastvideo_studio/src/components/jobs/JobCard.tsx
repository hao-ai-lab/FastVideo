'use client';

import * as React from 'react';
import { Timer } from 'lucide-react';

import CreateJobModal from '@/components/jobs/CreateJobModal';
import { Badge, type BadgeProps } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { useStore } from '@/hooks/useStore';
import {
  deleteJob,
  duplicateJob,
  downloadJobVideo,
  startJob,
  stopJob,
} from '@/lib/api';
import type { Job } from '@/lib/types';
import { cn, downloadBlob } from '@/lib/utils';
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

const BADGE_VARIANTS: Record<string, BadgeProps['variant']> = {
  pending: 'secondary',
  running: 'warning',
  completed: 'success',
  ready: 'success',
  failed: 'destructive',
  stopped: 'secondary',
  preprocessing: 'default',
};

export default function JobCard({ job, onJobUpdated }: JobCardProps) {
  const { activeJobId } = useStore(activeJobStore);
  const isSelected = activeJobId === job.id;

  const [isLoading, setIsLoading] = React.useState(false);
  const [isEditing, setIsEditing] = React.useState(false);
  const [isViewing, setIsViewing] = React.useState(false);
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

  async function handleDuplicate(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (isLoading) return;
    setIsLoading(true);
    try {
      await duplicateJob(job.id);
      onJobUpdated?.();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to duplicate job');
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

  function handleSelectJob() {
    setActiveJobId(isSelected ? null : job.id);
  }

  async function handleDownloadVideo(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (isLoading || !job.output_path) return;
    setIsLoading(true);
    try {
      const blob = await downloadJobVideo(job.id);
      const ext = job.output_path.endsWith('.png') ? 'png' : 'mp4';
      downloadBlob(blob, `job_${job.id}.${ext}`);
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to download video');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <article
      className={cn(
        'mb-3 flex cursor-pointer flex-col gap-2.5 rounded-lg border bg-background p-4 transition-colors last:mb-0',
        isSelected
          ? 'border-accent-blue bg-accent-blue/5'
          : 'border-border hover:border-muted-foreground/40',
      )}
    >
      <button
        type="button"
        aria-pressed={isSelected}
        onClick={handleSelectJob}
        className="flex w-full flex-col gap-2.5 rounded-md text-left"
      >
        <span className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-[0.95rem] font-semibold text-foreground">
            {job.name?.trim() || job.model_id}
          </span>
          <Badge variant={BADGE_VARIANTS[job.status] ?? 'secondary'}>
            {job.status}
          </Badge>
        </span>
        <span className="max-w-full overflow-hidden text-ellipsis whitespace-nowrap text-sm text-muted-foreground">
          {job.name?.trim() ? `${job.model_id} · ${job.prompt}` : job.prompt}
        </span>
        <span className="flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
          {/* Short job id; logs and output dirs are keyed on the full UUID. */}
          <span
            className="font-mono text-muted-foreground/80"
            title={job.id}
          >
            {job.id.slice(0, 8)}
          </span>
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
          {elapsedTime && (
            <span className="inline-flex items-center gap-1">
              <Timer className="size-3.5" aria-hidden />
              {elapsedTime}
            </span>
          )}
        </span>
      </button>
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
        {!(
          job.status === 'pending' ||
          job.status === 'failed' ||
          job.status === 'stopped'
        ) && (
          <Button
            size="sm"
            variant="outline"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setIsViewing(true);
            }}
            disabled={isLoading}
            title="View this job's configuration"
          >
            View
          </Button>
        )}
        {(job.status === 'pending' ||
          job.status === 'failed' ||
          job.status === 'stopped') && (
          <Button
            size="sm"
            variant="outline"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setIsEditing(true);
            }}
            disabled={isLoading}
            title="Edit this job's configuration"
          >
            Edit
          </Button>
        )}
        <Button
          size="sm"
          variant="outline"
          onClick={handleDuplicate}
          disabled={isLoading}
          title="Create a new pending job with this configuration"
        >
          Duplicate
        </Button>
        <Button
          size="sm"
          variant="destructive"
          onClick={handleDelete}
          disabled={isLoading}
        >
          Delete
        </Button>
      </div>
      {isViewing && (
        <CreateJobModal
          isOpen
          readOnly
          editingJob={job}
          jobType={(job.job_type ?? 'inference') as never}
          workloadType={job.workload_type ?? 't2v'}
          onClose={() => setIsViewing(false)}
          onSuccess={() => setIsViewing(false)}
        />
      )}
      {isEditing && (
        <CreateJobModal
          isOpen
          editingJob={job}
          jobType={(job.job_type ?? 'inference') as never}
          workloadType={job.workload_type ?? 't2v'}
          onClose={() => setIsEditing(false)}
          onSuccess={() => {
            setIsEditing(false);
            onJobUpdated?.();
          }}
        />
      )}
    </article>
  );
}
