'use client';

import { AlertTriangle, Loader2, RefreshCw } from 'lucide-react';
import * as React from 'react';

import CreateJobModal from '@/components/jobs/CreateJobModal';
import { EDITABLE_STATUSES, SceneBoard } from '@/components/scenes/SceneBoard';
import { MergedSceneDialog } from '@/components/scenes/MergedSceneDialog';
import { SceneQueueDialog } from '@/components/scenes/SceneQueueDialog';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { NativeSelect } from '@/components/ui/native-select';
import { toast } from 'sonner';
import { dequeueJob, getJobsList, mergeScene, queueJobs, updateJob, type MergedScene } from '@/lib/api';
import { attachedLastFrames, attachLastFrame, deferredLastFrameSource, detachLastFrame } from '@/lib/continuation';
import type { SceneQueuePlan } from '@/lib/sceneQueue';
import { buildScenes, type Scene, type SceneClip } from '@/lib/scenes';
import type { Job } from '@/lib/types';

const POLL_MS = 3000;

export default function ScenesPage() {
  const [jobs, setJobs] = React.useState<Job[]>([]);
  const [isLoading, setIsLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [reloadKey, setReloadKey] = React.useState(0);
  const [selectedKey, setSelectedKey] = React.useState<string | null>(null);
  const [openJob, setOpenJob] = React.useState<Job | null>(null);
  // which re-run of a clip number the user picked, by clip
  const [choices, setChoices] = React.useState<Record<string, string>>({});
  const [queueTarget, setQueueTarget] = React.useState<Scene | null>(null);
  const [merged, setMerged] = React.useState<(MergedScene & { title: string }) | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const list = await getJobsList('inference');
        if (cancelled) return;
        setJobs(list);
        setError(null);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load jobs');
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [reloadKey]);

  function reload() {
    setIsLoading(true);
    setReloadKey((k) => k + 1);
  }

  // While anything is queued or running, keep the board current without the
  // loading state (which would blank or flicker it).
  const hasActive = jobs.some((j) => j.status === 'queued' || j.status === 'running');
  React.useEffect(() => {
    if (!hasActive) return;
    const interval = setInterval(() => {
      getJobsList('inference')
        .then(setJobs)
        .catch(() => {});
    }, POLL_MS);
    return () => clearInterval(interval);
  }, [hasActive]);

  // Attach the previous clip's last frame to `clip` and rewrite its prompt to open on
  // it. The frame is a reference the server resolves when the clip runs, so the
  // previous clip doesn't have to be finished yet. Returns the picture label.
  async function attachPreviousFrame(clip: SceneClip, previous: SceneClip): Promise<string> {
    const result = attachLastFrame(
      clip.job.prompt,
      clip.job.references ?? [],
      deferredLastFrameSource(previous.job.id),
    );
    if (!result.ok) throw new Error(result.error);
    await updateJob(clip.job.id, { references: result.references, prompt: result.prompt });
    return result.label;
  }

  async function useLastFrame(clip: SceneClip, previous: SceneClip) {
    try {
      const label = await attachPreviousFrame(clip, previous);
      toast.success(`Clip ${clip.position} now starts from the last frame of clip ${previous.position} (${label}).`);
      reload();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't use the last frame.");
    }
  }

  // Undo the above: take every attached last frame back off the clip.
  async function stopUsingLastFrame(clip: SceneClip) {
    try {
      let prompt = clip.job.prompt;
      let references = clip.job.references ?? [];
      // Newest first: only the most recently added image can be removed cleanly.
      for (const { source } of attachedLastFrames(references).reverse()) {
        const result = detachLastFrame(prompt, references, source);
        if (!result.ok) throw new Error(result.error);
        prompt = result.prompt;
        references = result.references;
      }
      await updateJob(clip.job.id, { references, prompt });
      toast.success(`Clip ${clip.position} no longer starts from a previous clip's last frame.`);
      reload();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't remove the last frame.");
    }
  }

  // Join the scene's clips, in the order shown (the take chosen for each), into one video.
  async function mergeSceneClips(scene: Scene) {
    try {
      const result = await mergeScene(scene.clips.map((c) => c.job.id), scene.title);
      setMerged({ ...result, title: scene.title });
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't merge the scene.");
    }
  }

  async function queueClip(clip: SceneClip) {
    try {
      await queueJobs([clip.job.id]);
      toast.success(`Queued clip ${clip.position}.`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't queue the clip.");
    }
    reload();
  }

  async function dequeueClip(clip: SceneClip) {
    try {
      await dequeueJob(clip.job.id);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't remove the clip from the queue.");
    }
    reload();
  }

  // Link clips to the frames before them (if asked), then queue them all in scene order.
  async function queueScene(plan: SceneQueuePlan) {
    try {
      for (const { clip, previous } of plan.chain) {
        await attachPreviousFrame(clip, previous).catch((e) => {
          throw new Error(`Clip ${clip.position}: ${e instanceof Error ? e.message : e}. Nothing was queued.`);
        });
      }
      await queueJobs(plan.queue.map((c) => c.job.id));
      const n = plan.queue.length;
      toast.success(`Queued ${n} clip${n === 1 ? '' : 's'}.`);
      setQueueTarget(null);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't queue the scene.");
    }
    reload();
  }

  const scenes = React.useMemo(() => buildScenes(jobs, choices), [jobs, choices]);
  const scene = scenes.find((s) => s.key === selectedKey) ?? scenes[0];

  return (
    <div className="mx-auto w-full max-w-[1200px] px-4 pb-12 pt-4">
      <Card className="p-6">
        <div className="mb-1 flex flex-wrap items-center gap-3">
          <h2 className="text-2xl font-semibold text-foreground">Scenes</h2>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="ml-auto gap-1.5"
            onClick={reload}
            disabled={isLoading}
          >
            <RefreshCw className={isLoading ? 'size-3.5 animate-spin' : 'size-3.5'} aria-hidden />
            Refresh
          </Button>
        </div>
        <p className="mb-6 text-sm text-muted-foreground">
          Every shot across your jobs, in order. Jobs named with a shared stem and a number, like{' '}
          <code className="font-mono text-xs">wolf-lunch-clip-01</code>,{' '}
          <code className="font-mono text-xs">wolf-lunch-clip-02</code>, are grouped into one scene.
        </p>

        {isLoading && jobs.length === 0 ? (
          <div className="flex items-center gap-3 p-8 text-muted-foreground">
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
            <span>Loading scenes…</span>
          </div>
        ) : error && jobs.length === 0 ? (
          <div role="alert" className="flex flex-col items-center gap-3 py-8 text-center">
            <AlertTriangle className="size-6 text-destructive" aria-hidden />
            <p className="max-w-md text-sm text-muted-foreground">{error}</p>
            <Button type="button" variant="outline" onClick={reload}>
              Try Again
            </Button>
          </div>
        ) : !scene ? (
          <p className="py-8 text-center text-muted-foreground">
            No inference jobs yet. Create some and they&apos;ll show up here.
          </p>
        ) : (
          <div className="flex flex-col gap-6">
            {scenes.length > 1 && (
              <label className="flex max-w-md flex-col gap-1 text-xs text-muted-foreground">
                Scene
                <NativeSelect value={scene.key} onChange={(e) => setSelectedKey(e.target.value)}>
                  {scenes.map((s) => (
                    <option key={s.key} value={s.key}>
                      {s.title} ({s.clips.length} clip{s.clips.length === 1 ? '' : 's'})
                    </option>
                  ))}
                </NativeSelect>
              </label>
            )}
            <SceneBoard
              scene={scene}
              onOpenJob={setOpenJob}
              onUseLastFrame={useLastFrame}
              onDetachLastFrame={stopUsingLastFrame}
              onQueueScene={setQueueTarget}
              onMergeScene={mergeSceneClips}
              onQueueClip={queueClip}
              onDequeueClip={dequeueClip}
              onChooseTake={(takeKey, jobId) => setChoices((c) => ({ ...c, [takeKey]: jobId }))}
            />
          </div>
        )}
      </Card>

      <MergedSceneDialog merged={merged} onClose={() => setMerged(null)} />
      <SceneQueueDialog scene={queueTarget} onConfirm={queueScene} onClose={() => setQueueTarget(null)} />

      {openJob && (
        <CreateJobModal
          isOpen
          readOnly={!EDITABLE_STATUSES.includes(openJob.status)}
          editingJob={openJob}
          jobType={(openJob.job_type ?? 'inference') as never}
          workloadType={openJob.workload_type ?? 't2v'}
          onClose={() => setOpenJob(null)}
          onSuccess={() => {
            setOpenJob(null);
            reload();
          }}
        />
      )}
    </div>
  );
}
