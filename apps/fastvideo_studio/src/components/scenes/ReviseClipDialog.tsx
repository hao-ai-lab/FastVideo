'use client';

import * as React from 'react';
import { Loader2 } from 'lucide-react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import {
  duplicateJob,
  getClipRevision,
  startClipRevision,
  updateJob,
  type ClipRevision,
  type DraftedJob,
} from '@/lib/api';
import { formatDuration } from '@/lib/jobTiming';
import { diffLines } from '@/lib/lineDiff';
import type { SceneClip } from '@/lib/scenes';
import type { Job } from '@/lib/types';

const POLL_MS = 3000;

function seconds(job: Pick<DraftedJob, 'num_frames' | 'fps'>): string {
  return `${((job.num_frames ?? 0) / (job.fps || 24)).toFixed(1)} s`;
}

/** Changes outside the prompt text, as short sentences. */
function settingChanges(before: DraftedJob, after: DraftedJob): string[] {
  const changes: string[] = [];
  if (before.num_frames !== after.num_frames) {
    changes.push(`Length ${seconds(before)} → ${seconds(after)} (${before.num_frames} → ${after.num_frames} frames)`);
  }
  const sources = (j: DraftedJob) => (j.references ?? []).map((r) => r.source);
  const was = sources(before);
  const now = sources(after);
  for (const s of now.filter((s) => !was.includes(s))) changes.push(`Added reference ${s.split('/').pop()}`);
  for (const s of was.filter((s) => !now.includes(s))) changes.push(`Removed reference ${s.split('/').pop()}`);
  if (was.length === now.length && was.some((s, i) => s !== now[i])) changes.push('Reordered the references');
  if (before.seed !== after.seed) changes.push(`Seed ${before.seed} → ${after.seed}`);
  return changes;
}

export interface ReviseClipDialogProps {
  /** the clip to revise; its shown take is the one revised */
  clip: SceneClip | null;
  /** the shown takes of the clips before and after it, for continuity */
  neighbors: Job[];
  onClose: () => void;
  /** called after the revision is created as a new take */
  onCreated: () => void;
}

/**
 * Describe what went wrong in a clip's take; the scene writer revises the
 * clip's job, and the revision becomes a new pending take of the same clip
 * (same name), leaving the old take as it is.
 */
export function ReviseClipDialog({ clip, neighbors, onClose, onCreated }: ReviseClipDialogProps) {
  const [feedback, setFeedback] = React.useState('');
  const [revision, setRevision] = React.useState<ClipRevision | null>(null);
  const [starting, setStarting] = React.useState(false);
  const [creating, setCreating] = React.useState(false);
  const [now, setNow] = React.useState(() => Date.now());

  // Hiding the dialog keeps a running revision; opening it for a different take starts fresh.
  const jobId = clip?.job.id;
  const lastJobId = React.useRef<string | undefined>(undefined);
  React.useEffect(() => {
    if (jobId === undefined || jobId === lastJobId.current) return;
    lastJobId.current = jobId;
    setFeedback('');
    setRevision(null);
  }, [jobId]);

  const running = revision?.status === 'running';
  React.useEffect(() => {
    if (!running || !revision) return;
    const runId = revision.run_id;
    const poll = setInterval(async () => {
      try {
        setRevision(await getClipRevision(runId));
      } catch {
        // Keep the last state; the next poll retries.
      }
    }, POLL_MS);
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      clearInterval(poll);
      clearInterval(tick);
    };
  }, [running, revision?.run_id]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!clip) return null;
  const { job } = clip;

  async function start() {
    setStarting(true);
    try {
      setRevision(await startClipRevision({ job, feedback, neighbors }));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't start the revision.");
    } finally {
      setStarting(false);
    }
  }

  async function createTake() {
    if (!revision?.job) return;
    setCreating(true);
    try {
      const { id } = await duplicateJob(job.id);
      await updateJob(id, {
        prompt: revision.job.prompt.join('\n'),
        num_frames: revision.job.num_frames,
        references: revision.job.references ?? [],
        seed: revision.job.seed,
      });
      toast.success(`Created a new take of ${job.name}. It is pending; queue it when ready.`);
      onCreated();
      onClose();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't create the take.");
    } finally {
      setCreating(false);
    }
  }

  const done = revision?.status === 'completed' && revision.job !== null;
  const elapsed = revision ? (revision.finished_at ?? now / 1000) - revision.started_at : 0;
  const diff = done ? diffLines(revision.original.prompt.join('\n'), revision.job!.prompt.join('\n')) : [];
  const changes = done ? settingChanges(revision.original, revision.job!) : [];

  return (
    <Dialog open onOpenChange={(o) => !o && !creating && onClose()}>
      <DialogContent className="max-h-[90vh] max-w-3xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Revise {job.name}</DialogTitle>
          <DialogDescription>
            Describe what went wrong in this take. The scene writer finds the cause in the prompt and rewrites it; you
            review the changes, then create them as a new take of the same clip. The current take is kept.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-4 text-sm">
          <label className="flex flex-col gap-1">
            What went wrong
            <Textarea
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              rows={4}
              placeholder="e.g. There are 4 slaps instead of 2. / Andrew appears twice. / The band disappears after the chair throw."
              disabled={running}
            />
          </label>

          {revision && (
            <section aria-live="polite" className="flex flex-col gap-3 rounded-lg border border-border p-3">
              <div className="flex items-center gap-2">
                {running && <Loader2 className="size-4 animate-spin text-primary" aria-hidden />}
                <span className="font-medium">
                  {running ? 'Revising…' : done ? 'Revision ready' : 'Revision failed'}
                </span>
                <span className="text-xs text-muted-foreground">{formatDuration(elapsed)}</span>
              </div>
              {revision.error && <p className="text-xs text-destructive">{revision.error}</p>}
              {revision.final_response && (
                <p className="whitespace-pre-wrap text-xs">{revision.final_response}</p>
              )}
              {changes.length > 0 && (
                <ul className="list-disc pl-5 text-xs">
                  {changes.map((c) => (
                    <li key={c}>{c}</li>
                  ))}
                </ul>
              )}
              {done && (
                <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded bg-background p-2 text-[11px]">
                  {diff.map((line, i) => (
                    <div
                      key={i}
                      className={
                        line.kind === 'added'
                          ? 'bg-green-500/15 text-green-700 dark:text-green-400'
                          : line.kind === 'removed'
                            ? 'bg-red-500/15 text-red-700 line-through dark:text-red-400'
                            : 'text-muted-foreground'
                      }
                    >
                      {line.kind === 'added' ? '+ ' : line.kind === 'removed' ? '- ' : '  '}
                      {line.text}
                    </div>
                  ))}
                </pre>
              )}
            </section>
          )}
        </div>

        <DialogFooter>
          <Button type="button" variant="outline" onClick={onClose} disabled={creating}>
            {running ? 'Hide' : 'Cancel'}
          </Button>
          {done ? (
            <>
              <Button type="button" variant="outline" onClick={start} disabled={!feedback.trim() || creating}>
                Revise again
              </Button>
              <Button type="button" onClick={createTake} disabled={creating}>
                {creating ? 'Creating…' : 'Create new take'}
              </Button>
            </>
          ) : (
            <Button type="button" onClick={start} disabled={!feedback.trim() || running || starting}>
              {starting ? 'Starting…' : running ? 'Revising…' : 'Revise'}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
