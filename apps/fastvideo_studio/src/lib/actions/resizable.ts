// SPDX-License-Identifier: Apache-2.0
/**
 * Svelte action: attach horizontal resize behavior to a handle element.
 * Use on the resize-handle div inside a sidebar; the parent holds width state.
 */

export type ResizableEdge = "left" | "right";

export interface ResizableOptions {
	/** Which edge is fixed: "left" = drag right increases width, "right" = drag left increases width */
	edge: ResizableEdge;
	minWidth: number;
	maxWidth: number;
	/** Return current width (e.g. from parent state) */
	getWidth: () => number;
	/** Called with new width when user drags */
	onWidth: (width: number) => void;
	/** Optional: called when drag starts/stops so parent can show active state */
	onDragChange?: (dragging: boolean) => void;
}

export function resizable(
	node: HTMLElement,
	options: ResizableOptions,
): { update?: (opts: ResizableOptions) => void; destroy?: () => void } {
	let dragStart = { x: 0, width: 0 };
	let listenersAttached = false;

	function onMouseMove(e: MouseEvent) {
		const delta = e.clientX - dragStart.x;
		const sign = options.edge === "left" ? 1 : -1;
		const newWidth = Math.min(
			options.maxWidth,
			Math.max(options.minWidth, dragStart.width + sign * delta),
		);
		options.onWidth(newWidth);
	}

	function onMouseUp() {
		listenersAttached = false;
		options.onDragChange?.(false);
		document.body.style.cursor = "";
		document.body.style.userSelect = "";
		document.removeEventListener("mousemove", onMouseMove);
		document.removeEventListener("mouseup", onMouseUp);
	}

	function onMouseDown(e: MouseEvent) {
		e.preventDefault();
		dragStart = { x: e.clientX, width: options.getWidth() };
		listenersAttached = true;
		options.onDragChange?.(true);
		document.body.style.cursor = "col-resize";
		document.body.style.userSelect = "none";
		document.addEventListener("mousemove", onMouseMove);
		document.addEventListener("mouseup", onMouseUp);
	}

	node.addEventListener("mousedown", onMouseDown);

	return {
		update(newOptions: ResizableOptions) {
			options = newOptions;
		},
		destroy() {
			node.removeEventListener("mousedown", onMouseDown);
			if (listenersAttached) {
				options.onDragChange?.(false);
				document.body.style.cursor = "";
				document.body.style.userSelect = "";
				document.removeEventListener("mousemove", onMouseMove);
				document.removeEventListener("mouseup", onMouseUp);
			}
		},
	};
}
