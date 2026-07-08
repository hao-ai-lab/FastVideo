// SPDX-License-Identifier: Apache-2.0

import { writable } from "svelte/store";
import type { Job } from "$lib/types";

export const activeJobId = writable<string | null>(null);
export const activeJob = writable<Job | null>(null);

export function setActiveJobId(id: string | null): void {
	activeJobId.set(id);
	if (!id) activeJob.set(null);
}

export function setActiveJob(job: Job | null): void {
	activeJob.set(job);
}
