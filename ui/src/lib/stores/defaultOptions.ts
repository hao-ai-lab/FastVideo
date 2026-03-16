// SPDX-License-Identifier: Apache-2.0

import { writable } from "svelte/store";
import {
	DEFAULT_OPTIONS,
	loadDefaultOptions,
	saveDefaultOptions,
} from "$lib/defaultOptions";
import { getSettings, updateSettings } from "$lib/api";
import type { DefaultOptions } from "$lib/defaultOptions";

export const defaultOptions = writable<DefaultOptions>(loadDefaultOptions());

export function initDefaultOptions(): void {
	getSettings()
		.then((opts) => {
			// Merge with DEFAULT_OPTIONS so new fields (like apiServerBaseUrl)
			// always have sensible defaults even if the server is older.
			defaultOptions.set({ ...DEFAULT_OPTIONS, ...opts });
		})
		.catch(() => defaultOptions.set(loadDefaultOptions()));
}

export function updateOption<K extends keyof DefaultOptions>(
	key: K,
	value: DefaultOptions[K],
): void {
	defaultOptions.update((prev) => {
		const next = { ...prev, [key]: value };
		// API Server Base URL is a purely local (per-browser) setting.
		// Do not persist it to the backend; just update local storage.
		if (key === "apiServerBaseUrl") {
			saveDefaultOptions(next);
		} else {
			updateSettings({ [key]: value }).catch(() =>
				saveDefaultOptions(next),
			);
		}
		return next;
	});
}

export function resetToDefaults(): void {
	defaultOptions.set(DEFAULT_OPTIONS);
	updateSettings(DEFAULT_OPTIONS).catch(() =>
		saveDefaultOptions(DEFAULT_OPTIONS),
	);
}
