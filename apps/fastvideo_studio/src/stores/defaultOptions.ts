// SPDX-License-Identifier: Apache-2.0

import { getSettings, updateSettings } from '@/lib/api';
import {
  DEFAULT_OPTIONS,
  loadDefaultOptions,
  saveDefaultOptions,
  type DefaultOptions,
} from '@/lib/defaultOptions';
import { createManagedStore } from './createManagedStore';

export interface DefaultOptionsState {
  options: DefaultOptions;
}

export const defaultOptionsStore = createManagedStore<DefaultOptionsState>(
  { options: loadDefaultOptions() },
  // Server prerender and the hydration render must use the same deterministic
  // state (loadDefaultOptions reads localStorage on the client); the persisted
  // values take over right after hydration.
  { options: DEFAULT_OPTIONS },
);

export function initDefaultOptions(): void {
  getSettings()
    .then((opts) => {
      // Merge server settings into existing local options, but keep
      // apiServerBaseUrl purely local (do not let the server overwrite it).
      defaultOptionsStore.update((prev) => {
        const merged: DefaultOptions = {
          ...DEFAULT_OPTIONS,
          ...prev.options,
          ...opts,
          apiServerBaseUrl: prev.options.apiServerBaseUrl,
        };
        saveDefaultOptions(merged);
        return { options: merged };
      });
    })
    .catch(() => {
      // Fall back to whatever is in local storage (or DEFAULT_OPTIONS)
      defaultOptionsStore.set({ options: loadDefaultOptions() });
    });
}

export function updateOption<K extends keyof DefaultOptions>(
  key: K,
  value: DefaultOptions[K],
): void {
  defaultOptionsStore.update((prev) => {
    const next = { ...prev.options, [key]: value };
    // API Server Base URL is a purely local (per-browser) setting.
    // Do not persist it to the backend; just update local storage.
    if (key === 'apiServerBaseUrl') {
      saveDefaultOptions(next);
    } else {
      updateSettings({ [key]: value } as Partial<DefaultOptions>).catch(() =>
        saveDefaultOptions(next),
      );
    }
    return { options: next };
  });
}

export function resetToDefaults(): void {
  // apiServerBaseUrl is a purely local (per-browser) setting: keep the user's
  // value across a reset and never send it to the backend.
  const { apiServerBaseUrl: _local, ...serverDefaults } = DEFAULT_OPTIONS;
  const next: DefaultOptions = {
    ...DEFAULT_OPTIONS,
    apiServerBaseUrl: defaultOptionsStore.get().options.apiServerBaseUrl,
  };
  defaultOptionsStore.set({ options: next });
  updateSettings(serverDefaults).catch(() => saveDefaultOptions(next));
}
