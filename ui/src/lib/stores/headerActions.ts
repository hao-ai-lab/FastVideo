// SPDX-License-Identifier: Apache-2.0

import { writable } from "svelte/store";
import type { Component } from "svelte";

export interface HeaderAction {
	component: Component;
	props?: Record<string, unknown>;
}

/** Components to render in the header actions slot. */
export const headerActions = writable<HeaderAction[]>([]);

export function setHeaderActions(actions: HeaderAction[]): void {
	headerActions.set(actions);
}
