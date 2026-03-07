<script lang="ts">
	import type { Job } from "$lib/types";
	import {
		startJob,
		stopJob,
		deleteJob,
		downloadJobVideo,
	} from "$lib/api";
	import { activeJobId, setActiveJobId } from "$lib/stores/activeJob";

	let { job, onJobUpdated }: {
		job: Job;
		onJobUpdated?: () => void;
	} = $props();

	let isLoading = $state(false);
	let currentTime = $state(Date.now());

	const isSelected = $derived($activeJobId === job.id);
	const badgeClass =
		"badge" +
		job.status.charAt(0).toUpperCase() +
		job.status.slice(1);

	function formatDuration(seconds: number): string {
		const roundedSeconds = Math.round(seconds);
		if (roundedSeconds < 60) return `${roundedSeconds}s`;
		if (roundedSeconds < 3600) {
			const mins = Math.floor(roundedSeconds / 60);
			const secs = roundedSeconds % 60;
			return secs > 0 ? `${mins}m ${secs}s` : `${mins}m`;
		}
		const hours = Math.floor(roundedSeconds / 3600);
		const mins = Math.floor((roundedSeconds % 3600) / 60);
		return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
	}

	const elapsedTime = $derived((() => {
		if (!job.started_at) return null;
		const endTime =
			job.status === "running" ? currentTime : (job.finished_at ?? 0);
		if (!endTime && job.status !== "running") return null;
		const startedAtMs =
			job.started_at < 1e12 ? job.started_at * 1000 : job.started_at;
		const endTimeMs = endTime < 1e12 ? endTime * 1000 : endTime;
		const elapsedSeconds = (endTimeMs - startedAtMs) / 1000;
		if (elapsedSeconds <= 0) return null;
		return formatDuration(elapsedSeconds);
	})());

	$effect(() => {
		if (job.status !== "running" || !job.started_at) return;
		const interval = setInterval(() => (currentTime = Date.now()), 1000);
		return () => clearInterval(interval);
	});

	async function handleStart(e: MouseEvent) {
		e.preventDefault();
		e.stopPropagation();
		if (isLoading || job.status === "running" || job.status === "completed")
			return;
		isLoading = true;
		try {
			await startJob(job.id);
			onJobUpdated?.();
		} catch (err) {
			alert(err instanceof Error ? err.message : "Failed to start job");
		} finally {
			isLoading = false;
		}
	}

	async function handleStop(e: MouseEvent) {
		e.preventDefault();
		e.stopPropagation();
		if (isLoading || job.status !== "running") return;
		isLoading = true;
		try {
			await stopJob(job.id);
			onJobUpdated?.();
		} catch (err) {
			alert(err instanceof Error ? err.message : "Failed to stop job");
		} finally {
			isLoading = false;
		}
	}

	async function handleDelete(e: MouseEvent) {
		e.preventDefault();
		e.stopPropagation();
		if (isLoading) return;
		if (!confirm("Delete this job?")) return;
		isLoading = true;
		try {
			await deleteJob(job.id);
			onJobUpdated?.();
		} catch (err) {
			alert(err instanceof Error ? err.message : "Failed to delete job");
		} finally {
			isLoading = false;
		}
	}

	function handleSelectJob(e: MouseEvent) {
		if ((e.target as HTMLElement).closest("button")) return;
		setActiveJobId(isSelected ? null : job.id);
	}

	function handleKeyDown(e: KeyboardEvent) {
		if (e.key === "Enter" || e.key === " ") {
			e.preventDefault();
			handleSelectJob(e as unknown as MouseEvent);
		}
	}

	async function handleDownloadVideo(e: MouseEvent) {
		e.preventDefault();
		e.stopPropagation();
		if (isLoading || !job.output_path) return;
		isLoading = true;
		try {
			const blob = await downloadJobVideo(job.id);
			const ext = job.output_path.endsWith(".png") ? "png" : "mp4";
			const url = window.URL.createObjectURL(blob);
			const a = document.createElement("a");
			a.href = url;
			a.download = `job_${job.id}.${ext}`;
			document.body.appendChild(a);
			a.click();
			window.URL.revokeObjectURL(url);
			document.body.removeChild(a);
		} catch (err) {
			alert(
				err instanceof Error ? err.message : "Failed to download video",
			);
		} finally {
			isLoading = false;
		}
	}
</script>

<div
	class="jobCard"
	class:jobCardSelected={isSelected}
	onclick={handleSelectJob}
	onkeydown={handleKeyDown}
	role="button"
	tabindex="0"
>
	<div class="jobHeader">
		<span class="jobModel">{job.model_id}</span>
		<span class="badge {badgeClass}">{job.status}</span>
	</div>
	<p class="jobPrompt">{job.prompt}</p>
	<div class="jobMeta">
		{#if job.job_type === "inference"}
			<span>{job.num_frames} frames</span>
			<span>{job.height}×{job.width}</span>
		{:else}
			<span>{job.workload_type?.replace(/_/g, " ") ?? job.job_type}</span>
		{/if}
		{#if elapsedTime}
			<span class="jobDuration">⏱ {elapsedTime}</span>
		{/if}
	</div>
	<div class="jobActions">
		{#if job.status === "running"}
			<button class="btn btnStop btnSmall" onclick={handleStop} disabled={isLoading}>
				Stop
			</button>
		{:else if job.status === "failed"}
			<button class="btn btnStart btnSmall" onclick={handleStart} disabled={isLoading}>
				Restart
			</button>
		{:else if job.status === "pending" || job.status === "stopped"}
			<button class="btn btnStart btnSmall" onclick={handleStart} disabled={isLoading}>
				Start
			</button>
		{/if}
		{#if job.status === "completed" && job.output_path && job.job_type === "inference"}
			<button
				class="btn btnSmall"
				onclick={handleDownloadVideo}
				disabled={isLoading}
				title="Download video"
			>
				Download Video
			</button>
		{/if}
		<button class="btn btnDelete btnSmall" onclick={handleDelete} disabled={isLoading}>
			Delete
		</button>
	</div>
</div>

<style>
	.jobCard {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 1rem 1.15rem;
		margin-bottom: 0.75rem;
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
		cursor: pointer;
	}
	.jobCard:last-child {
		margin-bottom: 0;
	}
	.jobCardSelected {
		border-color: var(--accent);
		background: rgba(99, 102, 241, 0.06);
	}
	.jobHeader {
		display: flex;
		justify-content: space-between;
		align-items: center;
		flex-wrap: wrap;
		gap: 0.5rem;
	}
	.jobModel {
		font-weight: 600;
		font-size: 0.95rem;
	}
	.jobPrompt {
		font-size: 0.85rem;
		color: var(--text-dim);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		max-width: 100%;
	}
	.jobMeta {
		font-size: 0.75rem;
		color: var(--text-dim);
		display: flex;
		gap: 1rem;
		flex-wrap: wrap;
		align-items: center;
	}
	.jobActions {
		display: flex;
		gap: 0.4rem;
		flex-wrap: wrap;
		align-items: center;
	}
	.jobDuration {
		/* use default */
	}
	.badge {
		display: inline-block;
		font-size: 0.7rem;
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		padding: 0.15rem 0.5rem;
		border-radius: 999px;
	}
	.badgePending {
		background: var(--border);
		color: var(--text-dim);
	}
	.badgeRunning,
	.badgeCompleted,
	.badgeReady {
		background: rgba(34, 197, 94, 0.2);
		color: var(--green);
	}
	.badgeFailed {
		background: rgba(239, 68, 68, 0.2);
		color: var(--red);
	}
	.badgeStopped {
		background: rgba(156, 163, 175, 0.2);
		color: var(--text-dim);
	}
	.badgePreprocessing {
		background: rgba(99, 102, 241, 0.2);
		color: var(--accent);
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
		font-size: 0.75rem;
	}
	.btnStart {
		background: var(--green);
		color: #fff;
	}
	.btnStop {
		background: var(--yellow);
		color: #000;
	}
	.btnDelete {
		background: var(--red);
		color: #fff;
	}
</style>
