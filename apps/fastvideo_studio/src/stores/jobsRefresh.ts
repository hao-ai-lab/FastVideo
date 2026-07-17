// SPDX-License-Identifier: Apache-2.0

import { createManagedStore } from './createManagedStore';

export interface JobsRefreshState {
  nonce: number;
}

export const jobsRefreshStore = createManagedStore<JobsRefreshState>({
  nonce: 0,
});

/** Bump the nonce so subscribers (e.g. JobQueue) refetch. */
export function triggerRefresh(): void {
  jobsRefreshStore.update((state) => ({ nonce: state.nonce + 1 }));
}
