'use client';

import { AlertTriangle, ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';
import { useEffect, useState } from 'react';

import JobFilters from '@/components/jobs/JobFilters';
import JobResultPreview, { JobResultMetadata } from '@/components/jobs/JobResultPreview';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { useJobModelOptions } from '@/hooks/useJobModelOptions';
import { getJobsList } from '@/lib/api';
import { filterJobs, hasJobResult, MAX_RESULT_PREVIEWS } from '@/lib/jobResults';
import type { Job } from '@/lib/types';

export default function GalleryPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [modelFilter, setModelFilter] = useState('');
  const [promptFilter, setPromptFilter] = useState('');
  const [page, setPage] = useState(0);
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

  const galleryJobs = jobs.filter(hasJobResult);
  const modelOptions = useJobModelOptions(galleryJobs);
  const filteredJobs = filterJobs(galleryJobs, modelFilter, promptFilter);
  const pageCount = Math.ceil(filteredJobs.length / MAX_RESULT_PREVIEWS);
  const currentPage = Math.min(page, Math.max(0, pageCount - 1));
  const pageJobs = filteredJobs.slice(
    currentPage * MAX_RESULT_PREVIEWS,
    (currentPage + 1) * MAX_RESULT_PREVIEWS,
  );

  return (
    <div className="mx-auto w-full max-w-[1200px] px-4 pb-12 pt-4">
      <Card className="p-6">
        <h2 className="mb-1 text-2xl font-semibold text-foreground">Gallery</h2>
        <p className="mb-6 text-sm text-muted-foreground">
          Completed videos and images, with their model and settings.
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
          <>
            <JobFilters
              model={modelFilter}
              models={modelOptions}
              prompt={promptFilter}
              onModelChange={(value) => { setModelFilter(value); setPage(0); }}
              onPromptChange={(value) => { setPromptFilter(value); setPage(0); }}
              count={filteredJobs.length}
              total={galleryJobs.length}
            />
            {filteredJobs.length === 0 ? (
              <p className="py-8 text-center text-muted-foreground">No results match these filters.</p>
            ) : (
              <div className="grid grid-cols-[repeat(auto-fill,minmax(min(100%,320px),1fr))] gap-5">
                {pageJobs.map((job) => (
                  <article
                    key={job.id}
                    className="flex flex-col overflow-hidden rounded-lg border border-border bg-background"
                  >
                    <JobResultPreview job={job} className="rounded-none" />
                    <div className="flex flex-col gap-2.5 border-t border-border p-4">
                      <div>
                        <h3 className="truncate text-sm font-semibold" title={job.name?.trim() || job.model_id}>
                          {job.name?.trim() || job.model_id}
                        </h3>
                        {job.name?.trim() && <p className="mt-1 truncate text-xs text-muted-foreground" title={job.model_id}>{job.model_id}</p>}
                      </div>
                      <p className="line-clamp-3 text-sm text-muted-foreground" title={job.prompt}>
                        {job.prompt || '—'}
                      </p>
                      <JobResultMetadata job={job} />
                    </div>
                  </article>
                ))}
              </div>
            )}
            {pageCount > 1 && (
              <nav aria-label="Gallery pages" className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-border pt-4">
                <p className="text-xs text-muted-foreground">
                  Page {currentPage + 1} of {pageCount} · Up to {MAX_RESULT_PREVIEWS} previews per page
                </p>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>
                    <ChevronLeft className="mr-1 size-4" aria-hidden /> Previous
                  </Button>
                  <Button variant="outline" size="sm" disabled={currentPage + 1 >= pageCount} onClick={() => setPage(currentPage + 1)}>
                    Next <ChevronRight className="ml-1 size-4" aria-hidden />
                  </Button>
                </div>
              </nav>
            )}
          </>
        )}
      </Card>
    </div>
  );
}
