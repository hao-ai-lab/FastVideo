<script lang="ts">
	import { onMount } from "svelte";
	import { getJobsList } from "$lib/api";
	import { getJobVideoUrl } from "$lib/api";
	import type { Job } from "$lib/types";

	let jobs = $state<Job[]>([]);
	let isLoading = $state(true);
	let error = $state<string | null>(null);

	const galleryJobs = $derived(
		jobs.filter(
			(j) =>
				j.status === "completed" &&
				j.output_path &&
				(j.job_type === "inference" || !j.job_type),
		),
	);

	async function fetchJobs() {
		try {
			error = null;
			const list = await getJobsList("inference");
			jobs = list.sort(
				(a, b) =>
					(b.finished_at ?? b.created_at ?? 0) -
					(a.finished_at ?? a.created_at ?? 0),
			);
		} catch (e) {
			error = e instanceof Error ? e.message : "Failed to load jobs";
		} finally {
			isLoading = false;
		}
	}

	onMount(() => {
		fetchJobs();
	});

	function isImage(job: Job): boolean {
		return job.output_path?.toLowerCase().endsWith(".png") ?? false;
	}
</script>

<main class="main">
	<section class="card">
		<h2 class="pageTitle">Gallery</h2>
		<p class="pageDesc">
			Generated videos from completed inference jobs. Captions show the prompt
			used for each generation.
		</p>
		{#if isLoading}
			<div class="loading">
				<div class="spinner"></div>
				<span>Loading gallery…</span>
			</div>
		{:else if error}
			<p class="error">{error}</p>
		{:else if galleryJobs.length === 0}
			<p class="placeholder">
				No generated videos yet. Complete some
				<a href="/inference">inference jobs</a> to see them here.
			</p>
		{:else}
			<div class="grid">
				{#each galleryJobs as job (job.id)}
					<article class="tile">
						<div class="mediaWrap">
							{#if isImage(job)}
								<img
									src={getJobVideoUrl(job.id)}
									alt={job.prompt}
									class="media"
									loading="lazy"
								/>
							{:else}
								<video
									src={getJobVideoUrl(job.id)}
									class="media"
									muted
									loop
									playsinline
									preload="metadata"
								></video>
							{/if}
						</div>
						<p class="caption" title={job.prompt}>{job.prompt || "—"}</p>
					</article>
				{/each}
			</div>
		{/if}
	</section>
</main>

<style>
	.main {
		max-width: 1200px;
		margin: 0 auto;
		padding: 0 1rem 3rem;
		width: 100%;
	}
	.card {
		padding: 1.5rem;
	}
	.pageTitle {
		font-size: 1.5rem;
		font-weight: 600;
		margin: 0 0 0.25rem 0;
		color: var(--text);
	}
	.pageDesc {
		font-size: 0.9rem;
		color: var(--text-dim);
		margin: 0 0 1.5rem 0;
	}
	.loading {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 2rem;
		color: var(--text-dim);
	}
	.spinner {
		width: 24px;
		height: 24px;
		border: 2px solid var(--border);
		border-top-color: var(--accent);
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
	.error {
		color: var(--red);
		padding: 2rem 0;
	}
	.placeholder {
		text-align: center;
		color: var(--text-dim);
		padding: 2rem 0;
	}
	.placeholder a {
		color: var(--accent);
		text-decoration: none;
	}
	.placeholder a:hover {
		text-decoration: underline;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
		gap: 1.25rem;
	}
	.tile {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		overflow: hidden;
		display: flex;
		flex-direction: column;
	}
	.mediaWrap {
		aspect-ratio: 16 / 9;
		background: var(--surface);
		position: relative;
		overflow: hidden;
	}
	.media {
		width: 100%;
		height: 100%;
		object-fit: contain;
		display: block;
	}
	.caption {
		font-size: 0.85rem;
		color: var(--text-dim);
		padding: 0.75rem 1rem;
		margin: 0;
		line-height: 1.4;
		display: -webkit-box;
		-webkit-line-clamp: 3;
		line-clamp: 3;
		-webkit-box-orient: vertical;
		overflow: hidden;
		border-top: 1px solid var(--border);
	}
</style>
