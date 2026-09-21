'use client';

import { useEffect, useMemo, useState } from 'react';

import { getModels, type Model } from '@/lib/api';
import type { Job } from '@/lib/types';

/** Share the create-job catalog while retaining historical/custom job models. */
export function useJobModelOptions(jobs: Job[], workloadType?: string): Model[] {
  const [catalog, setCatalog] = useState<Model[]>([]);

  useEffect(() => {
    let stale = false;
    setCatalog([]);
    getModels(workloadType)
      .then((models) => {
        if (!stale) setCatalog(models);
      })
      .catch(() => {
        // Recorded jobs remain filterable if the catalog is unavailable.
        if (!stale) setCatalog([]);
      });
    return () => {
      stale = true;
    };
  }, [workloadType]);

  return useMemo(() => {
    const models = new Map(catalog.map((model) => [model.id, model]));
    for (const job of jobs) {
      if (job.model_id && !models.has(job.model_id)) {
        models.set(job.model_id, { id: job.model_id, label: job.model_id });
      }
    }
    return [...models.values()];
  }, [catalog, jobs]);
}
