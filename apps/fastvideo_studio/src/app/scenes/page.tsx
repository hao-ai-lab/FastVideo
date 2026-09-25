'use client';

import { AlertTriangle, Loader2, RefreshCw } from 'lucide-react';
import * as React from 'react';

import CreateJobModal from '@/components/jobs/CreateJobModal';
import { EDITABLE_STATUSES, SceneBoard } from '@/components/scenes/SceneBoard';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { NativeSelect } from '@/components/ui/native-select';
import { getJobsList } from '@/lib/api';
import { buildScenes } from '@/lib/scenes';
import type { Job } from '@/lib/types';

export default function ScenesPage() {
  const [jobs, setJobs] = React.useState<Job[]>([]);
  const [isLoading, setIsLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [reloadKey, setReloadKey] = React.useState(0);
  const [selectedKey, setSelectedKey] = React.useState<string | null>(null);
  const [openJob, setOpenJob] = React.useState<Job | null>(null);
  // which re-run of a clip number the user picked, by clip
  const [choices, setChoices] = React.useState<Record<string, string>>({});

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
              onChooseTake={(takeKey, jobId) => setChoices((c) => ({ ...c, [takeKey]: jobId }))}
            />
          </div>
        )}
      </Card>

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
