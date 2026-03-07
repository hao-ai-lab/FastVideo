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
		.then((opts) => defaultOptions.set(opts))
		.catch(() => defaultOptions.set(loadDefaultOptions()));
}

export function updateOption<K extends keyof DefaultOptions>(
	key: K,
	value: DefaultOptions[K],
): void {
	defaultOptions.update((prev) => {
		const next = { ...prev, [key]: value };
		updateSettings({ [key]: value }).catch(() => saveDefaultOptions(next));
		return next;
	});
}

export function resetToDefaults(): void {
	defaultOptions.set(DEFAULT_OPTIONS);
	updateSettings(DEFAULT_OPTIONS).catch(() =>
		saveDefaultOptions(DEFAULT_OPTIONS),
	);
}
