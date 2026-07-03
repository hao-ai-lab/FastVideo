<script lang="ts">
	import { onMount, onDestroy } from "svelte";
	import { getJobsList } from "$lib/api";
	import type { JobType } from "$lib/types";
	import type { Job } from "$lib/types";
	import JobCard from "./JobCard.svelte";
	import { activeJobId, setActiveJobId, setActiveJob } from "$lib/stores/activeJob";
	import { registerRefresh, triggerRefresh } from "$lib/stores/jobsRefresh";

	let {
		jobType,
		jobTypesForList,
	}: { jobType: JobType; jobTypesForList?: JobType[] } = $props();

	let jobs = $state<Job[]>([]);
	let interval: ReturnType<typeof setInterval> | null = null;

	const typesToFetch = $derived(jobTypesForList ?? [jobType]);

	const activeJob = $derived(
		$activeJobId ? jobs.find((j) => j.id === $activeJobId) ?? null : null,
	);

	$effect(() => {
		setActiveJob(activeJob);
		// Void active job ID if deleeted
		if ($activeJobId && !activeJob) setActiveJobId(null);
	});

	async function fetchJobs() {
		try {
			if (typesToFetch.length === 1) {
				jobs = await getJobsList(typesToFetch[0]);
			} else {
				const results = await Promise.all(
					typesToFetch.map((t) => getJobsList(t)),
				);
				jobs = results
					.flat()
					.sort(
						(a, b) =>
							new Date(b.created_at ?? 0).getTime() -
							new Date(a.created_at ?? 0).getTime(),
					);
			}
		} catch (e) {
			console.error("Failed to fetch jobs:", e);
		}
	}

	$effect(() => {
		const hasActive = jobs.some(
			(j) => j.status === "running" || j.status === "pending",
		);
		if (hasActive) {
			interval = setInterval(fetchJobs, 1000);
		} else if (interval) {
			clearInterval(interval);
			interval = null;
		}
		return () => {
			if (interval) clearInterval(interval);
		};
	});

	onMount(() => {
		fetchJobs();
		return registerRefresh(fetchJobs);
	});

	onDestroy(() => {
		if (interval) clearInterval(interval);
	});
</script>

<main class="main">
	<section class="card">
		<div id="jobs-container">
			{#if jobs.length === 0}
				<p class="placeholder">
					No {typesToFetch.length > 1 ? "jobs" : jobType + " jobs"} yet. Create
					one above.
				</p>
			{:else}
				{#each jobs as job (job.id)}
					<JobCard {job} onJobUpdated={fetchJobs} />
				{/each}
			{/if}
		</div>
	</section>
</main>

<style>
	.main {
		max-width: 850px;
		margin: 0 auto;
		padding: 0 1rem 3rem;
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
		width: 100%;
	}
	.card {
		padding: 1.5rem;
	}
	.placeholder {
		text-align: center;
		color: var(--text-dim);
		padding: 2rem 0;
	}
</style>
