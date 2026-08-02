'use client';

import { AlertTriangle, ImageOff, Loader2 } from 'lucide-react';
import { useEffect, useState } from 'react';

import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { getJobVideoUrl, getJobsList } from '@/lib/api';
import type { Job } from '@/lib/types';

function isImage(job: Job): boolean {
  return job.output_path?.toLowerCase().endsWith('.png') ?? false;
}

function GalleryMedia({ job }: { job: Job }) {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <div
        role="status"
        className="flex h-full flex-col items-center justify-center gap-2 px-4 text-center text-muted-foreground"
      >
        <ImageOff className="size-7" aria-hidden />
        <span className="text-sm font-medium">Preview unavailable</span>
        <span className="text-xs">
          The generated file could not be loaded.
        </span>
      </div>
    );
  }

  if (isImage(job)) {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={getJobVideoUrl(job.id)}
        alt={job.prompt}
        className="block h-full w-full object-contain"
        loading="lazy"
        onError={() => setFailed(true)}
      />
    );
  }

  return (
    <video
      src={getJobVideoUrl(job.id)}
      aria-label={
        job.prompt ? `Generated video: ${job.prompt}` : 'Generated video'
      }
      className="block h-full w-full object-contain"
      controls
      muted
      loop
      playsInline
      preload="metadata"
      onError={() => setFailed(true)}
    />
  );
}

export default function GalleryPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const list = await getJobsList('inference');
        if (cancelled) return;
        const sorted = [...list].sort(
          (a, b) =>
            (b.finished_at ?? b.created_at ?? 0) -
            (a.finished_at ?? a.created_at ?? 0),
        );
        setJobs(sorted);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Failed to load jobs');
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [reloadKey]);

  function retry() {
    setError(null);
    setIsLoading(true);
    setReloadKey((k) => k + 1);
  }

  const galleryJobs = jobs.filter(
    (j) =>
      j.status === 'completed' &&
      j.output_path &&
      (j.job_type === 'inference' || !j.job_type),
  );

  return (
    <div className="mx-auto w-full max-w-[1200px] px-4 pb-12 pt-4">
      <Card className="p-6">
        <h2 className="mb-1 text-2xl font-semibold text-foreground">Gallery</h2>
        <p className="mb-6 text-sm text-muted-foreground">
          Generated videos from completed inference jobs. Captions show the
          prompt used for each generation.
        </p>

        {isLoading ? (
          <div className="flex items-center gap-3 p-8 text-muted-foreground">
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
            <span>Loading gallery…</span>
          </div>
        ) : error ? (
          <div
            role="alert"
            className="flex flex-col items-center gap-3 py-8 text-center"
          >
            <AlertTriangle className="size-6 text-destructive" aria-hidden />
            <p className="max-w-md text-sm text-muted-foreground">{error}</p>
            <Button type="button" variant="outline" onClick={retry}>
              Try Again
            </Button>
          </div>
        ) : galleryJobs.length === 0 ? (
          <p className="py-8 text-center text-muted-foreground">
            No completed videos yet
          </p>
        ) : (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-5">
            {galleryJobs.map((job) => (
              <article
                key={job.id}
                className="flex flex-col overflow-hidden rounded-lg border border-border bg-background"
              >
                <div className="relative aspect-video overflow-hidden bg-muted">
                  <GalleryMedia job={job} />
                </div>
                <p
                  className="line-clamp-3 border-t border-border px-4 py-3 text-sm text-muted-foreground"
                  title={job.prompt}
                >
                  {job.prompt || '—'}
                </p>
              </article>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
