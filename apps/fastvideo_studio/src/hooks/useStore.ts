'use client';

import { useSyncExternalStore } from 'react';

import type { ManagedStore } from '@/stores/createManagedStore';

export function useStore<T>(
  store: ManagedStore<T>,
): T {
  return useSyncExternalStore(store.subscribe, store.get, store.get);
}
