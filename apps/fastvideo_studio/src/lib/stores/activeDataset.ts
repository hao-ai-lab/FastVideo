// SPDX-License-Identifier: Apache-2.0

import { writable } from "svelte/store";
import type { Dataset } from "$lib/api";

export const activeDatasetId = writable<string | null>(null);
export const activeDataset = writable<Dataset | null>(null);

export function setActiveDatasetId(id: string | null): void {
	activeDatasetId.set(id);
	if (!id) activeDataset.set(null);
}

export function setActiveDataset(dataset: Dataset | null): void {
	activeDataset.set(dataset);
}
