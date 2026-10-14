'use client';

import * as React from 'react';

import { computeElapsed } from '@/lib/jobTiming';
import type { Job } from '@/lib/types';

/** A job's formatted run time, ticking every second while it runs. */
export function useJobElapsed(job: Job): string | null {
  const [currentTime, setCurrentTime] = React.useState(() => Date.now());

  React.useEffect(() => {
    if (job.status !== 'running' || !job.started_at) return;
    const interval = setInterval(() => setCurrentTime(Date.now()), 1000);
    return () => clearInterval(interval);
  }, [job.status, job.started_at]);

  return computeElapsed(job, currentTime);
}
