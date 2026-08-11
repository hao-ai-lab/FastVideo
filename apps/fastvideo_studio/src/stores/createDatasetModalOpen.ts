// SPDX-License-Identifier: Apache-2.0

import { createManagedStore } from './createManagedStore';

export interface CreateDatasetModalState {
  open: boolean;
}

export const createDatasetModalStore =
  createManagedStore<CreateDatasetModalState>({ open: false });

export function setCreateDatasetModalOpen(open: boolean): void {
  createDatasetModalStore.patch({ open });
}
