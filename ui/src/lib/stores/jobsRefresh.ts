// SPDX-License-Identifier: Apache-2.0

type RefreshCallback = () => void;

let refreshCallback: RefreshCallback | null = null;

export function registerRefresh(cb: RefreshCallback): () => void {
	refreshCallback = cb;
	return () => {
		refreshCallback = null;
	};
}

export function triggerRefresh(): void {
	refreshCallback?.();
}
