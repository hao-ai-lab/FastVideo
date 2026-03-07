<script lang="ts">
	import type { Job } from "$lib/types";
	import { getJobLogs, downloadJobLog } from "$lib/api";
	import { resizable } from "$lib/actions/resizable";

	let { job, onClose, onWidthChange }: {
		job: Job;
		onClose: () => void;
		onWidthChange?: (w: number) => void;
	} = $props();

	const SIDEBAR_MIN_WIDTH = 280;
	const SIDEBAR_MAX_WIDTH = 750;

	let width = $state(360);
	let isDragging = $state(false);
	let isLoading = $state(false);
	let logs = $state<string[]>([]);
	let logAfter = $state(0);
	let consoleEl: HTMLPreElement;
	let previousJobId = $state<string | null>(null);
	let previousStatus = $state<string | null>(null);
	const stateRef = { logs: [] as string[], logAfter: 0 };
	let pollingLock = false;

	$effect(() => {
		onWidthChange?.(width);
	});

	$effect(() => {
		const wasTerminal =
			previousStatus === "failed" ||
			previousStatus === "stopped" ||
			previousStatus === "completed";
		const isRestarting =
			previousJobId === job.id &&
			wasTerminal &&
			(job.status === "pending" || job.status === "running");

		if (previousJobId !== job.id || isRestarting) {
			logs = [];
			logAfter = 0;
			stateRef.logs = [];
			stateRef.logAfter = 0;
		}
		previousJobId = job.id;
		previousStatus = job.status;
	});

	$effect(() => {
		const shouldPoll = job.status === "running" || job.status === "pending";
		let pollInterval: ReturnType<typeof setInterval> | null = null;
		let mounted = true;

		async function pollLogs() {
			if (!mounted || pollingLock) return;
			pollingLock = true;
			try {
				const logData = await getJobLogs(job.id, stateRef.logAfter);
				if (mounted && logData.lines.length > 0) {
					stateRef.logs = [...stateRef.logs, ...logData.lines];
					stateRef.logAfter = logData.total;
					logs = stateRef.logs;
					logAfter = stateRef.logAfter;
					if (consoleEl) consoleEl.scrollTop = consoleEl.scrollHeight;
				}
			} catch (e) {
				console.error("Failed to fetch logs:", e);
			} finally {
				pollingLock = false;
			}
		}

		pollLogs();
		if (shouldPoll) pollInterval = setInterval(pollLogs, 2000);

		return () => {
			mounted = false;
			if (pollInterval) clearInterval(pollInterval);
		};
	});

	const resizableOpts = $derived({
		edge: "right" as const,
		minWidth: SIDEBAR_MIN_WIDTH,
		maxWidth: SIDEBAR_MAX_WIDTH,
		getWidth: () => width,
		onWidth: (w: number) => (width = w),
		onDragChange: (dragging: boolean) => (isDragging = dragging),
	});

	async function handleDownloadLog(e: MouseEvent) {
		e.preventDefault();
		if (isLoading) return;
		isLoading = true;
		try {
			const blob = await downloadJobLog(job.id);
			const url = window.URL.createObjectURL(blob);
			const a = document.createElement("a");
			a.href = url;
			a.download = `job_${job.id}.log`;
			document.body.appendChild(a);
			a.click();
			window.URL.revokeObjectURL(url);
			document.body.removeChild(a);
		} catch (err) {
			console.error("Failed to download log:", err);
			alert(err instanceof Error ? err.message : "Failed to download log");
		} finally {
			isLoading = false;
		}
	}
</script>

<aside class="sidebar" style="width: {width}px; max-width: {SIDEBAR_MAX_WIDTH}px">
	<div class="header">
		<h2 class="title">Job Details</h2>
		<div class="headerActions">
			<button
				class="btn btnSmall"
				onclick={handleDownloadLog}
				disabled={isLoading || !job.log_file_path}
				title="Download log file"
			>
				Download Log
			</button>
			<button type="button" class="closeBtn" onclick={onClose} title="Close">
				<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
					<path d="M18 6L6 18M6 6l12 12" />
				</svg>
			</button>
		</div>
	</div>
	<div class="consoleSection">
		<div class="consoleHeader">
			<span class="consoleTitle">Console Output</span>
			{#if job.status === "running"}
				<span class="consoleStatus">● Live</span>
			{/if}
		</div>
		<pre bind:this={consoleEl} class="consoleOutput">{#if logs.length === 0}<span class="consoleEmpty">{job.status === "running" ? "Waiting for logs..." : "No logs available"}</span>{:else}{logs.join("\n")}{/if}</pre>
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
		min-width: 280px;
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
	.consoleSection {
		flex: 1;
		display: flex;
		flex-direction: column;
		min-height: 0;
		padding: 1rem 1.25rem;
	}
	.consoleHeader {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.5rem;
	}
	.consoleTitle {
		font-size: 0.8rem;
		font-weight: 600;
		color: var(--text-dim);
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}
	.consoleStatus {
		font-size: 0.7rem;
		color: var(--green);
		font-weight: 500;
	}
	.consoleOutput {
		flex: 1;
		min-height: 0;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 0.75rem;
		font-family: Monaco, Menlo, "Ubuntu Mono", Consolas, monospace;
		font-size: 0.75rem;
		line-height: 1.5;
		color: var(--text);
		overflow-y: auto;
		overflow-x: auto;
		white-space: pre-wrap;
		word-wrap: break-word;
		margin: 0;
	}
	.consoleEmpty {
		color: var(--text-dim);
		font-style: italic;
	}
	.btn {
		cursor: pointer;
		border: none;
		border-radius: var(--radius);
		font-family: inherit;
		font-weight: 600;
		font-size: 0.85rem;
		padding: 0.55rem 1.2rem;
		transition: background 0.15s, opacity 0.15s;
	}
	.btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.btnSmall {
		background: transparent;
		color: var(--accent-h);
		border: 1px solid var(--border);
		padding: 0.3rem 0.7rem;
		font-size: 0.8rem;
	}
	.btnSmall:hover {
		background: var(--border);
	}
</style>
