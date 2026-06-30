// SPDX-License-Identifier: Apache-2.0

import type { Dataset } from '@/lib/api';
import { createManagedStore } from './createManagedStore';

export interface ActiveDatasetState {
  activeDatasetId: string | null;
  activeDataset: Dataset | null;
}

export const activeDatasetStore = createManagedStore<ActiveDatasetState>({
  activeDatasetId: null,
  activeDataset: null,
});

export function setActiveDatasetId(id: string | null): void {
  activeDatasetStore.patch(
    id
      ? { activeDatasetId: id }
      : { activeDatasetId: null, activeDataset: null },
  );
}

export function setActiveDataset(dataset: Dataset | null): void {
  activeDatasetStore.patch({ activeDataset: dataset });
}
