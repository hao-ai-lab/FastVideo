<script lang="ts">
	import type { Dataset } from "$lib/api";
	import {
		getDatasetFiles,
		updateDatasetCaption,
		getDatasetMediaUrl,
	} from "$lib/api";
	import DownloadCaptions from "./DownloadCaptions.svelte";
	import { resizable } from "$lib/actions/resizable";

	let { dataset, onClose, onWidthChange }: {
		dataset: Dataset;
		onClose: () => void;
		onWidthChange?: (w: number) => void;
	} = $props();

	const SIDEBAR_MIN_WIDTH = 320;
	const SIDEBAR_MAX_WIDTH = 900;
	const INITIAL_PAGE_SIZE = 24;
	const PAGE_SIZE = 24;

	let width = $state(400);
	let isDragging = $state(false);
	let fileNames = $state<string[]>([]);
	let captions = $state<Record<string, string>>({});
	let visibleCount = $state(INITIAL_PAGE_SIZE);
	let isLoading = $state(true);
	let thumbLoaded = $state<Record<string, boolean>>({});
	let saveTimeout: ReturnType<typeof setTimeout> | null = null;
	let scrollEl: HTMLDivElement;

	$effect(() => {
		onWidthChange?.(width);
	});

	$effect(() => {
		let cancelled = false;
		isLoading = true;
		getDatasetFiles(dataset.id)
			.then((data) => {
				if (!cancelled) {
					fileNames = data.file_names;
					captions = data.captions;
					visibleCount = INITIAL_PAGE_SIZE;
					thumbLoaded = {};
				}
			})
			.catch((err) => console.error("Failed to load dataset files:", err))
			.finally(() => {
				if (!cancelled) isLoading = false;
			});
		return () => {
			cancelled = true;
		};
	});

	function handleCaptionChange(fileName: string, value: string) {
		captions = { ...captions, [fileName]: value };
		if (saveTimeout) clearTimeout(saveTimeout);
		saveTimeout = setTimeout(() => {
			saveTimeout = null;
			updateDatasetCaption(dataset.id, fileName, value).catch((err) =>
				console.error("Failed to save caption:", err),
			);
		}, 500);
	}

	function handleLoadMore() {
		visibleCount = Math.min(visibleCount + PAGE_SIZE, fileNames.length);
	}

	const resizableOpts = $derived({
		edge: "right" as const,
		minWidth: SIDEBAR_MIN_WIDTH,
		maxWidth: SIDEBAR_MAX_WIDTH,
		getWidth: () => width,
		onWidth: (w: number) => (width = w),
		onDragChange: (dragging: boolean) => (isDragging = dragging),
	});

	$effect(() => {
		return () => {
			if (saveTimeout) clearTimeout(saveTimeout);
		};
	});

	const visibleFiles = $derived(fileNames.slice(0, visibleCount));
	const hasMore = $derived(visibleCount < fileNames.length);

	function handleScroll() {
		if (!hasMore || isLoading || !scrollEl) return;
		const { scrollTop, scrollHeight, clientHeight } = scrollEl;
		const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
		if (distanceFromBottom < 200) {
			visibleCount = Math.min(visibleCount + PAGE_SIZE, fileNames.length);
		}
	}
</script>

<aside class="sidebar" style="width: {width}px; max-width: {SIDEBAR_MAX_WIDTH}px">
	<div class="header">
		<h2 class="title">{dataset.name}</h2>
		<div class="headerActions">
			<DownloadCaptions {fileNames} {captions} />
			<button
				type="button"
				class="closeBtn"
				onclick={onClose}
				title="Close"
			>
				<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
					<path d="M18 6L6 18M6 6l12 12" />
				</svg>
			</button>
		</div>
	</div>
	<div class="gallerySection">
		<div
			class="galleryScroll"
			bind:this={scrollEl}
			onscroll={handleScroll}
		>
			{#if isLoading}
				<p class="galleryEmpty">Loading…</p>
			{:else if fileNames.length === 0}
				<p class="galleryEmpty">No media files</p>
			{:else}
				<div class="galleryGrid">
					{#each visibleFiles as fileName}
						<div class="galleryItem">
							{#if !thumbLoaded[fileName]}
								<div class="thumbLoading">
									<div class="thumbSpinner"></div>
								</div>
							{/if}
							<video
								src={getDatasetMediaUrl(dataset.id, fileName)}
								class="galleryThumb"
								muted
								autoplay
								loop
								playsinline
								onloadeddata={() => {
									thumbLoaded = { ...thumbLoaded, [fileName]: true };
								}}
								onerror={() => {
									thumbLoaded = { ...thumbLoaded, [fileName]: true };
								}}
							></video>
							<div class="galleryCaption">
								<textarea
									value={captions[fileName] ?? ""}
									oninput={(e) =>
										handleCaptionChange(
											fileName,
											(e.target as HTMLTextAreaElement).value,
										)}
									placeholder="Caption"
									rows={2}
								></textarea>
							</div>
						</div>
					{/each}
				</div>
			{/if}
		</div>
	</div>
	<div
		class="resizeHandle"
		class:resizeHandleActive={isDragging}
		use:resizable={resizableOpts}
		role="presentation"
	></div>
</aside>

<style>
	.sidebar {
		position: fixed;
		top: var(--header-height);
		right: 0;
		bottom: 0;
		display: flex;
		flex-direction: column;
		max-height: calc(100vh - var(--header-height));
		min-width: 320px;
		background: var(--surface);
		border-left: 1px solid var(--border);
		flex-shrink: 0;
		z-index: 50;
	}
	.resizeHandle {
		position: absolute;
		top: 0;
		left: 0;
		width: 6px;
		bottom: 0;
		cursor: col-resize;
		z-index: 1;
	}
	.resizeHandle:hover,
	.resizeHandleActive {
		background: rgba(99, 102, 241, 0.2);
	}
	.header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 1rem 1.25rem;
		border-bottom: 1px solid var(--border);
		flex-shrink: 0;
	}
	.title {
		font-size: 1rem;
		font-weight: 600;
		margin: 0;
		color: var(--text);
	}
	.headerActions {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	.closeBtn {
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 0.4rem;
		background: none;
		border: none;
		color: var(--text-dim);
		cursor: pointer;
		border-radius: var(--radius);
		transition: color 0.15s, background 0.15s;
	}
	.closeBtn:hover {
		color: var(--text);
		background: rgba(99, 102, 241, 0.08);
	}
	.closeBtn svg {
		width: 18px;
		height: 18px;
	}
	.gallerySection {
		flex: 1;
		display: flex;
		flex-direction: column;
		min-height: 0;
		overflow: hidden;
	}
	.galleryScroll {
		flex: 1;
		overflow-y: auto;
		padding: 1rem;
	}
	.galleryGrid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
		gap: 1rem;
	}
	.galleryItem {
		display: flex;
		flex-direction: column;
		position: relative;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		overflow: hidden;
	}
	.galleryThumb {
		aspect-ratio: 16 / 9;
		object-fit: cover;
		width: 100%;
		background: var(--border);
	}
	.galleryCaption {
		font-size: 0.8rem;
	}
	.galleryCaption textarea {
		width: 100%;
		min-height: 2.5rem;
		padding: 0.4rem;
		font-size: 0.8rem;
		font-family: inherit;
		border: none;
		background: transparent;
		color: var(--text);
		resize: vertical;
	}
	.thumbLoading {
		position: absolute;
		inset: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		background: radial-gradient(
			circle at center,
			rgba(15, 17, 23, 0.6),
			rgba(15, 17, 23, 0.9)
		);
		pointer-events: none;
	}
	.thumbSpinner {
		width: 24px;
		height: 24px;
		border-radius: 999px;
		border: 2px solid rgba(148, 163, 184, 0.4);
		border-top-color: var(--accent);
		animation: thumb-spin 0.7s linear infinite;
	}
	@keyframes thumb-spin {
		to {
			transform: rotate(360deg);
		}
	}
	.galleryEmpty {
		color: var(--text-dim);
		text-align: center;
		padding: 2rem;
	}
</style>
