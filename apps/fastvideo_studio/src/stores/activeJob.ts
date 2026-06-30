// SPDX-License-Identifier: Apache-2.0

import type { Job } from '@/lib/types';
import { createManagedStore } from './createManagedStore';

export interface ActiveJobState {
  activeJobId: string | null;
  activeJob: Job | null;
}

export const activeJobStore = createManagedStore<ActiveJobState>({
  activeJobId: null,
  activeJob: null,
});

export function setActiveJobId(id: string | null): void {
  activeJobStore.patch(
    id ? { activeJobId: id } : { activeJobId: null, activeJob: null },
  );
}

export function setActiveJob(job: Job | null): void {
  activeJobStore.patch({ activeJob: job });
}
