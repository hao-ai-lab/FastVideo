'use client';

import * as React from 'react';

import JobCard from '@/components/JobCard';
import { useStore } from '@/hooks/useStore';
import { getJobsList } from '@/lib/api';
import type { Job, JobType } from '@/lib/types';
import {
  activeJobStore,
  setActiveJob,
  setActiveJobId,
} from '@/stores/activeJob';
import { jobsRefreshStore } from '@/stores/jobsRefresh';

interface JobQueueProps {
  jobType: JobType;
  jobTypesForList?: JobType[];
}

// Jobs are flat objects of primitives, so a shallow compare detects "nothing
// changed" across poll responses (which are referentially fresh every fetch).
function jobsShallowEqual(a: Job | null, b: Job | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  const aKeys = Object.keys(a) as (keyof Job)[];
  return (
    aKeys.length === Object.keys(b).length &&
    aKeys.every((k) => a[k] === b[k])
  );
}

export default function JobQueue({ jobType, jobTypesForList }: JobQueueProps) {
  const [jobs, setJobs] = React.useState<Job[]>([]);
  const { nonce } = useStore(jobsRefreshStore);
  const { activeJobId } = useStore(activeJobStore);

  // Stable primitive key so the memoized list below keeps its identity when an
  // inline array prop (e.g. ['finetuning', 'lora']) gets a fresh reference each
  // render, which would otherwise re-run the fetch effects needlessly.
  const typesKey = (jobTypesForList ?? [jobType]).join(',');
  const typesToFetch = React.useMemo<JobType[]>(
    () => jobTypesForList ?? [jobType],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [typesKey],
  );

  // Guard the poll against slow responses: `inFlight` lets the interval skip
  // a tick instead of stacking requests, and the sequence counter drops
  // out-of-order responses so a delayed older payload can't overwrite a newer
  // job list (direct refetches always run and supersede in-flight polls).
  const fetchSeq = React.useRef(0);
  const inFlight = React.useRef(false);

  const fetchJobs = React.useCallback(async () => {
    const seq = ++fetchSeq.current;
    inFlight.current = true;
    try {
      let next: Job[];
      if (typesToFetch.length === 1) {
        next = await getJobsList(typesToFetch[0]);
      } else {
        const results = await Promise.all(
          typesToFetch.map((t) => getJobsList(t)),
        );
        next = results
          .flat()
          .sort(
            (a, b) =>
              new Date(b.created_at ?? 0).getTime() -
              new Date(a.created_at ?? 0).getTime(),
          );
      }
      if (seq === fetchSeq.current) setJobs(next);
    } catch (e) {
      console.error('Failed to fetch jobs:', e);
    } finally {
      if (seq === fetchSeq.current) inFlight.current = false;
    }
  }, [typesKey]);

  // Keep the latest fetchJobs reachable from the polling interval without
  // tearing it down/recreating it on every fetchJobs identity change.
  const fetchJobsRef = React.useRef(fetchJobs);
  React.useEffect(() => {
    fetchJobsRef.current = fetchJobs;
  }, [fetchJobs]);

  // Initial fetch + refetch whenever an external refresh is triggered.
  React.useEffect(() => {
    fetchJobs();
  }, [fetchJobs, nonce]);

  // Poll every second while any job is running/pending; stop otherwise.
  const hasActive = jobs.some(
    (j) => j.status === 'running' || j.status === 'pending',
  );
  React.useEffect(() => {
    if (!hasActive) return;
    const interval = setInterval(() => {
      if (!inFlight.current) fetchJobsRef.current();
    }, 1000);
    return () => clearInterval(interval);
  }, [hasActive]);

  // Keep the active job in sync with the selected id, and void the selection
  // if its job disappeared (e.g. deleted). Skip the store write when nothing
  // changed — each poll returns fresh objects, and an unconditional write
  // would re-render every subscriber (shell, sidebars, cards) once a second.
  React.useEffect(() => {
    if (!activeJobId) {
      if (activeJobStore.get().activeJob) setActiveJob(null);
      return;
    }
    const activeJob = jobs.find((j) => j.id === activeJobId) ?? null;
    if (!jobsShallowEqual(activeJob, activeJobStore.get().activeJob)) {
      setActiveJob(activeJob);
    }
    if (!activeJob) setActiveJobId(null);
  }, [activeJobId, jobs]);

  const multiType = typesToFetch.length > 1;

  return (
    <main className="mx-auto flex w-full max-w-[850px] flex-col gap-6 px-4 pb-12">
      <section className="p-6">
        <div>
          {jobs.length === 0 ? (
            <p className="py-8 text-center text-muted-foreground">
              No {multiType ? 'jobs' : `${jobType} jobs`} yet. Create one above.
            </p>
          ) : (
            jobs.map((job) => (
              <JobCard key={job.id} job={job} onJobUpdated={fetchJobs} />
            ))
          )}
        </div>
      </section>
    </main>
  );
}
