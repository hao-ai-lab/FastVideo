<script lang="ts">
	import { page } from "$app/stores";
	import { resizable } from "$lib/actions/resizable";

	const SIDEBAR_MIN_WIDTH = 100;
	const SIDEBAR_MAX_WIDTH = 300;
	const SIDEBAR_COLLAPSED_WIDTH = 0;
	const SIDEBAR_COLLAPSED_VISIBLE_WIDTH = 60;

	let { onWidthChange }: { onWidthChange?: (w: number) => void } = $props();

	let width = $state(220);
	let isCollapsed = $state(false);
	let isDragging = $state(false);

	const effectiveWidth = isCollapsed ? SIDEBAR_COLLAPSED_WIDTH : width;
	const layoutWidth = isCollapsed ? SIDEBAR_COLLAPSED_VISIBLE_WIDTH : width;

	$effect(() => {
		onWidthChange?.(layoutWidth);
	});

	const resizableOpts = $derived({
		edge: "left" as const,
		minWidth: SIDEBAR_MIN_WIDTH,
		maxWidth: SIDEBAR_MAX_WIDTH,
		getWidth: () => width,
		onWidth: (w: number) => (width = w),
		onDragChange: (dragging: boolean) => (isDragging = dragging),
	});

	function toggleCollapse() {
		isCollapsed = !isCollapsed;
	}
</script>

<aside
	class="sidebar"
	class:collapsed={isCollapsed}
	style="width: {effectiveWidth}px"
>
	<nav class="tabs">
		<a href="/inference" class="tab" class:tabActive={$page.url.pathname === "/inference"}>Inference</a>
		<a href="/finetuning" class="tab" class:tabActive={$page.url.pathname === "/finetuning"}>Finetuning</a>
		<a href="/distillation" class="tab" class:tabActive={$page.url.pathname === "/distillation"}>Distillation</a>
		<a href="/lora" class="tab" class:tabActive={$page.url.pathname === "/lora"}>LoRA</a>
		<a href="/datasets" class="tab" class:tabActive={$page.url.pathname === "/datasets"}>Datasets</a>
		<a href="/settings" class="tab" class:tabActive={$page.url.pathname === "/settings"}>Settings</a>
	</nav>
	<div class="collapseFooter">
		<button
			type="button"
			class="collapseBtn"
			onclick={toggleCollapse}
			title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
		>
			{#if isCollapsed}
				<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
					<path d="M9 18l6-6-6-6" />
				</svg>
			{:else}
				<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
					<path d="M15 18l-6-6 6-6" />
				</svg>
			{/if}
		</button>
	</div>
	{#if !isCollapsed}
		<div
			class="resizeHandle"
			class:resizeHandleActive={isDragging}
			use:resizable={resizableOpts}
			role="presentation"
		></div>
	{/if}
</aside>

<style>
	.sidebar {
		position: fixed;
		top: var(--header-height);
		left: 0;
		bottom: 0;
		display: flex;
		flex-direction: column;
		max-height: calc(100vh - var(--header-height));
		background: var(--surface);
		border-right: 1px solid var(--border);
		flex-shrink: 0;
		z-index: 50;
	}
	.resizeHandle {
		position: absolute;
		top: 0;
		right: 0;
		width: 6px;
		bottom: 0;
		cursor: col-resize;
		z-index: 1;
	}
	.resizeHandle:hover,
	.resizeHandleActive {
		background: rgba(99, 102, 241, 0.2);
	}
	.resizeHandle::after {
		content: "";
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		width: 2px;
		height: 24px;
		background: var(--border);
		border-radius: 1px;
		opacity: 0;
		transition: opacity 0.15s;
	}
	.resizeHandle:hover::after,
	.resizeHandleActive::after {
		opacity: 1;
	}
	.collapseFooter {
		padding: 0.5rem;
		position: absolute;
		bottom: 0;
		right: 0;
	}
	.collapseBtn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 100%;
		padding: 0.5rem;
		background: none;
		border: none;
		color: var(--text-dim);
		cursor: pointer;
		border-radius: var(--radius);
		font-family: inherit;
		transition: color 0.15s, background 0.15s;
	}
	.collapseBtn:hover {
		color: var(--text);
		background: rgba(99, 102, 241, 0.08);
	}
	.collapseBtn svg {
		width: 18px;
		height: 18px;
	}
	.collapsed .tabs {
		display: none;
	}
	.collapsed .collapseFooter {
		border-top: none;
		top: 0;
		right: -60px;
	}
	.collapsed .collapseBtn {
		width: auto;
		padding: 0.75rem;
	}
	.tabs {
		display: flex;
		flex-direction: column;
		padding: 0.5rem 0;
	}
	.tab {
		display: block;
		padding: 0.65rem 1.25rem;
		font-size: 0.9rem;
		color: var(--text-dim);
		text-decoration: none;
		border: none;
		background: none;
		text-align: left;
		font-family: inherit;
		transition: color 0.15s, background 0.15s;
	}
	.tab:hover {
		color: var(--text);
		background: rgba(99, 102, 241, 0.08);
	}
	.tabActive {
		color: var(--accent);
		font-weight: 500;
		background: rgba(99, 102, 241, 0.12);
	}
</style>
