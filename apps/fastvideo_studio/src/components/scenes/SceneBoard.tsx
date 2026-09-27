'use client';

import * as React from 'react';

import { H3ShotPanel } from '@/components/jobs/H3ShotPanel';
import { Badge, type BadgeProps } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { NativeSelect } from '@/components/ui/native-select';
import { getJobVideoUrl } from '@/lib/api';
import { attachedLastFrames, whyCannotAttach } from '@/lib/continuation';
import { formatCutTime } from '@/lib/h3Shots';
import { QUEUEABLE_STATUSES } from '@/lib/sceneQueue';
import { formatClock, type Scene, type SceneClip } from '@/lib/scenes';
import type { Job } from '@/lib/types';
import { cn } from '@/lib/utils';

const BADGE_VARIANTS: Record<string, BadgeProps['variant']> = {
  pending: 'secondary',
  queued: 'default',
  running: 'warning',
  completed: 'success',
  failed: 'destructive',
  stopped: 'secondary',
};

const SEGMENT_TONE: Record<string, string> = {
  completed: 'bg-emerald-500/70',
  running: 'bg-amber-500/70',
  queued: 'bg-sky-500/60',
  failed: 'bg-rose-500/70',
};

/** Statuses a job can still be edited in (mirrors the Jobs page). */
export const EDITABLE_STATUSES = ['pending', 'failed', 'stopped'];

function clipLabel(clip: SceneClip): string {
  return clip.job.name?.trim() || `Job ${clip.job.id.slice(0, 8)}`;
}

function takeLabel(job: Job, index: number, total: number): string {
  const ms = job.created_at < 1e12 ? job.created_at * 1000 : job.created_at;
  const when = new Date(ms).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
  return `Take ${index + 1} of ${total} · ${job.status} · ${when}`;
}

function scrollToClip(id: string) {
  document.getElementById(`clip-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function Timeline({ scene }: { scene: Scene }) {
  return (
    <div className="flex h-4 w-full gap-px overflow-hidden rounded-md" role="group" aria-label="Scene timeline">
      {scene.clips.map((clip) => (
        <button
          key={clip.job.id}
          type="button"
          onClick={() => scrollToClip(clip.job.id)}
          aria-label={`Go to clip ${clip.position}: ${clipLabel(clip)}`}
          title={`Clip ${clip.position} · ${clipLabel(clip)} · ${clip.job.status} · ${formatClock(clip.startSeconds)}–${formatClock(clip.startSeconds + clip.durationSeconds)}`}
          style={{ flexGrow: Math.max(clip.durationSeconds, 0.1), flexBasis: 0 }}
          className={cn(
            'min-w-[3px] transition-opacity hover:opacity-70',
            SEGMENT_TONE[clip.job.status] ?? 'bg-muted-foreground/40',
          )}
        />
      ))}
    </div>
  );
}

function ClipPreview({ job }: { job: Job }) {
  const isImage = job.output_path?.toLowerCase().endsWith('.png') ?? false;
  const src = getJobVideoUrl(job.id);
  return isImage ? (
    // eslint-disable-next-line @next/next/no-img-element
    <img src={src} alt={job.name ?? 'Generated output'} className="max-h-56 rounded-md border border-border" />
  ) : (
    <video
      src={src}
      aria-label={`Generated video for ${job.name ?? job.id}`}
      className="max-h-56 rounded-md border border-border bg-black"
      controls
      muted
      loop
      playsInline
      preload="metadata"
    />
  );
}

function ClipRow({
  clip,
  previous,
  firstShotNumber,
  onOpenJob,
  onChooseTake,
  onUseLastFrame,
  onDetachLastFrame,
  onQueueClip,
  onDequeueClip,
}: {
  clip: SceneClip;
  /** the clip before this one in the scene, if any */
  previous: SceneClip | undefined;
  firstShotNumber: number;
  onOpenJob: (job: Job) => void;
  onChooseTake: (takeKey: string, jobId: string) => void;
  onUseLastFrame: (clip: SceneClip, previous: SceneClip) => Promise<void>;
  onDetachLastFrame: (clip: SceneClip) => Promise<void>;
  onQueueClip: (clip: SceneClip) => Promise<void>;
  onDequeueClip: (clip: SceneClip) => Promise<void>;
}) {
  const [previewing, setPreviewing] = React.useState(false);
  const [attaching, setAttaching] = React.useState(false);
  const [queueing, setQueueing] = React.useState(false);
  const { job } = clip;
  const canPreview = job.status === 'completed' && !!job.output_path;
  const editable = EDITABLE_STATUSES.includes(job.status);

  const queued = job.status === 'queued';
  const canQueue = QUEUEABLE_STATUSES.includes(job.status);

  // Starting from the previous clip's last frame: offered on clips that can still be
  // edited. The frame is taken when this clip runs, so the previous clip needn't be done yet.
  const frames = attachedLastFrames(job.references);
  const fromPrevious = previous && frames.some((f) => f.jobId === previous.job.id);
  const fromAnotherTake = frames.some((f) => f.jobId !== previous?.job.id);
  const previousDone = !!previous && previous.job.status === 'completed';
  const cannotUseFrame = previous ? whyCannotAttach(job.prompt, job.references) : null;

  const usingLastFrame = frames.length > 0;
  const lastFrameBlocked = attaching || (!usingLastFrame && !!cannotUseFrame);

  async function toggleLastFrame() {
    if (!previous) return;
    setAttaching(true);
    try {
      await (usingLastFrame ? onDetachLastFrame(clip) : onUseLastFrame(clip, previous));
    } finally {
      setAttaching(false);
    }
  }

  async function queueOrDequeue(action: (clip: SceneClip) => Promise<void>) {
    setQueueing(true);
    try {
      await action(clip);
    } finally {
      setQueueing(false);
    }
  }

  return (
    <li id={`clip-${job.id}`} className="scroll-mt-32 rounded-xl border border-border bg-card p-3">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="flex size-6 shrink-0 items-center justify-center rounded-md bg-secondary text-xs font-semibold text-foreground">
          {clip.position}
        </span>
        <h3 className="min-w-0 truncate text-sm font-semibold text-foreground">{clipLabel(clip)}</h3>
        <Badge variant={BADGE_VARIANTS[job.status] ?? 'secondary'}>{job.status}</Badge>
        <span className="font-mono text-xs text-muted-foreground">
          {formatClock(clip.startSeconds)}–{formatClock(clip.startSeconds + clip.durationSeconds)}
        </span>
        {clip.continuation && (
          <Badge variant="outline" className="normal-case tracking-normal">
            continues previous shot
          </Badge>
        )}
        <span className="ml-auto flex flex-wrap items-center justify-end gap-2">
          {previous && queued && fromPrevious && (
            <Badge variant="outline" className="normal-case tracking-normal">
              starts from clip {previous.position}&apos;s last frame
            </Badge>
          )}
          {previous && queued && fromPrevious && !previousDone && (
            <Badge variant="outline" className="normal-case tracking-normal">
              waiting for clip {previous.position}
            </Badge>
          )}
          {previous && editable && (
            <label
              className={cn(
                'flex items-center gap-1.5 text-xs text-foreground',
                lastFrameBlocked && 'text-muted-foreground',
              )}
              title={
                usingLastFrame
                  ? "Stop starting this clip from the previous clip's last frame"
                  : (cannotUseFrame ??
                    `Attach the last frame of clip ${previous.position} as this clip's opening frame${
                      previousDone ? '' : ` (taken when this clip runs, after clip ${previous.position} has finished)`
                    }`)
              }
            >
              <input
                type="checkbox"
                checked={usingLastFrame}
                onChange={toggleLastFrame}
                disabled={lastFrameBlocked}
                className="size-4"
              />
              {attaching
                ? usingLastFrame
                  ? 'Removing…'
                  : 'Attaching…'
                : fromPrevious || !usingLastFrame
                  ? `Use last frame of clip ${previous.position}`
                  : "Starts from another take's last frame"}
            </label>
          )}
          {clip.takeKey && clip.takes.length > 1 && (
            <NativeSelect
              aria-label={`Take for clip ${clip.position}`}
              value={job.id}
              onChange={(e) => onChooseTake(clip.takeKey as string, e.target.value)}
              className="h-8 w-auto py-0 text-xs"
            >
              {clip.takes.map((take, i) => (
                <option key={take.id} value={take.id}>
                  {takeLabel(take, i, clip.takes.length)}
                </option>
              ))}
            </NativeSelect>
          )}
          {canPreview && (
            <Button type="button" size="sm" variant="outline" onClick={() => setPreviewing((p) => !p)}>
              {previewing ? 'Hide video' : 'Preview'}
            </Button>
          )}
          {canQueue && (
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => queueOrDequeue(onQueueClip)}
              disabled={queueing}
              title="Run this clip when the ones queued before it have finished"
            >
              Queue
            </Button>
          )}
          {queued && (
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => queueOrDequeue(onDequeueClip)}
              disabled={queueing}
            >
              Remove from queue
            </Button>
          )}
          <Button type="button" size="sm" variant="outline" onClick={() => onOpenJob(job)}>
            {editable ? 'Edit' : 'View'}
          </Button>
        </span>
      </div>

      {previewing && canPreview && (
        <div className="mb-2">
          <ClipPreview job={job} />
        </div>
      )}

      {clip.shots.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border p-3 text-sm text-muted-foreground">
          This job&apos;s prompt has no shots.
        </p>
      ) : (
        <div className="flex items-stretch gap-2 overflow-x-auto pb-1" aria-label={`Shots in ${clipLabel(clip)}`}>
          {clip.shots.map((shot, i) => (
            <div
              key={shot.id}
              className="flex w-52 shrink-0 flex-col gap-1.5 rounded-xl border border-border bg-background p-2"
            >
              <H3ShotPanel
                shot={shot}
                number={firstShotNumber + i}
                timeLabel={formatCutTime(clip.shotStarts[i])}
              />
            </div>
          ))}
        </div>
      )}
    </li>
  );
}

export interface SceneBoardProps {
  scene: Scene;
  onOpenJob: (job: Job) => void;
  /** start a clip from the last frame of the one before it; resolves when done (errors are reported by the caller) */
  onUseLastFrame: (clip: SceneClip, previous: SceneClip) => Promise<void>;
  /** pick which re-run of a clip number to show */
  onChooseTake: (takeKey: string, jobId: string) => void;
  /** open the confirmation for queueing every runnable clip of the scene */
  onQueueScene: (scene: Scene) => void;
  /** join the scene's finished clips into one video; resolves when done (errors are reported by the caller) */
  onMergeScene: (scene: Scene) => Promise<void>;
  /** stop starting a clip from a previous clip's last frame; resolves when done */
  onDetachLastFrame: (clip: SceneClip) => Promise<void>;
  /** queue / unqueue one clip; resolve when done (errors are reported by the caller) */
  onQueueClip: (clip: SceneClip) => Promise<void>;
  onDequeueClip: (clip: SceneClip) => Promise<void>;
}

/** One scene: a timeline of its clips, then every clip's shots in order, numbered straight through. */
export function SceneBoard({
  scene,
  onOpenJob,
  onChooseTake,
  onUseLastFrame,
  onDetachLastFrame,
  onQueueScene,
  onMergeScene,
  onQueueClip,
  onDequeueClip,
}: SceneBoardProps) {
  let shotNumber = 1;
  const statuses = Object.entries(scene.statusCounts);
  const anythingToQueue = scene.clips.some((c) => QUEUEABLE_STATUSES.includes(c.job.status));

  // Merging needs every clip's video, so a scene missing one can't pass for the whole thing.
  const unfinished = scene.clips.filter((c) => c.job.status !== 'completed' || !c.job.output_path).length;
  const [merging, setMerging] = React.useState(false);
  async function merge() {
    setMerging(true);
    try {
      await onMergeScene(scene);
    } finally {
      setMerging(false);
    }
  }

  // The header stays where it is until it reaches the top of the scrolling area,
  // then sticks there. A 1px sentinel sitting just above it tells us which: once
  // the sentinel has scrolled out of view, the header is stuck.
  const sentinelRef = React.useRef<HTMLDivElement>(null);
  const [stuck, setStuck] = React.useState(false);
  React.useEffect(() => {
    const el = sentinelRef.current;
    if (!el || typeof IntersectionObserver === 'undefined') return;
    const observer = new IntersectionObserver(([entry]) => setStuck(!entry.isIntersecting));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <section aria-label={scene.title} className="flex flex-col gap-4">
      <div ref={sentinelRef} aria-hidden className="-mb-4 h-px" />
      <div
        data-stuck={stuck}
        className={cn(
          'sticky top-0 z-30 -mx-3 flex flex-col gap-2 bg-card px-3 pb-3 pt-2 transition-shadow',
          stuck && 'border-b border-border shadow-md',
        )}
      >
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <h2 className="text-xl font-semibold text-foreground">{scene.title}</h2>
          <p className="text-sm text-muted-foreground">
            {scene.clips.length} clip{scene.clips.length === 1 ? '' : 's'} · {scene.shotCount} shot
            {scene.shotCount === 1 ? '' : 's'} · about {formatClock(scene.totalSeconds)}
          </p>
          <span className="flex flex-wrap gap-1.5">
            {statuses.map(([status, count]) => (
              <Badge key={status} variant={BADGE_VARIANTS[status] ?? 'secondary'}>
                {count} {status}
              </Badge>
            ))}
          </span>
          <span className="ml-auto flex flex-wrap gap-2">
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={merge}
              disabled={unfinished > 0 || merging}
              title={
                unfinished > 0
                  ? `${unfinished} clip${unfinished === 1 ? " hasn't" : "s haven't"} finished yet`
                  : 'Join all the clips, in order, into one video'
              }
            >
              {merging ? 'Merging…' : 'Merge scene'}
            </Button>
            <Button
              type="button"
              size="sm"
              onClick={() => onQueueScene(scene)}
              disabled={!anythingToQueue}
              title={anythingToQueue ? 'Queue every clip that has not run yet' : 'Every clip has already run or is queued'}
            >
              Queue scene
            </Button>
          </span>
        </div>
        <Timeline scene={scene} />
      </div>

      <ol className="flex list-none flex-col gap-3 p-0">
        {scene.clips.map((clip, index) => {
          const first = shotNumber;
          shotNumber += clip.shots.length;
          return (
            <ClipRow
              key={clip.takeKey ?? clip.job.id}
              clip={clip}
              previous={scene.clips[index - 1]}
              firstShotNumber={first}
              onOpenJob={onOpenJob}
              onChooseTake={onChooseTake}
              onUseLastFrame={onUseLastFrame}
              onDetachLastFrame={onDetachLastFrame}
              onQueueClip={onQueueClip}
              onDequeueClip={onDequeueClip}
            />
          );
        })}
      </ol>
    </section>
  );
}
