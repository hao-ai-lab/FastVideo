<script lang="ts">
	import "../app.css";
	import { onMount } from "svelte";
	import { page } from "$app/stores";
	import Header from "$lib/components/Header.svelte";
	import PrimarySidebar from "$lib/components/PrimarySidebar.svelte";
	import SecondarySidebar from "$lib/components/SecondarySidebar.svelte";
	import DatasetSidebar from "$lib/components/DatasetSidebar.svelte";
	import { initDefaultOptions } from "$lib/stores/defaultOptions";
	import { activeJob, setActiveJobId } from "$lib/stores/activeJob";
	import { activeDataset, setActiveDatasetId } from "$lib/stores/activeDataset";

	onMount(() => {
		initDefaultOptions();
	});

	let primaryWidth = $state(220);
	let secondaryWidth = $state(0);

	const jobRoutes = ["/inference", "/finetuning", "/distillation", "/lora"];
	const pathname = $derived($page.url.pathname);
	const jobSidebarOpen = $derived(
		jobRoutes.includes(pathname) && $activeJob != null,
	);
	const datasetSidebarOpen = $derived(
		pathname === "/datasets" && $activeDataset != null,
	);
	const secondaryOpen = $derived(jobSidebarOpen || datasetSidebarOpen);

	function handleKeyDown(e: KeyboardEvent) {
		if (e.key === "Escape" && !document.querySelector("[data-modal]")) {
			if ($activeJob) setActiveJobId(null);
			if ($activeDataset) setActiveDatasetId(null);
		}
	}

	onMount(() => {
		document.addEventListener("keydown", handleKeyDown);
		return () => document.removeEventListener("keydown", handleKeyDown);
	});
</script>

<Header />
<div class="layout">
	<PrimarySidebar onWidthChange={(w) => (primaryWidth = w)} />
	<div
		class="content"
		style="margin-left: {primaryWidth}px; margin-right: {secondaryOpen ? secondaryWidth : 0}px;"
	>
		<slot />
	</div>
	{#if jobSidebarOpen && $activeJob}
		<SecondarySidebar
			job={$activeJob}
			onClose={() => setActiveJobId(null)}
			onWidthChange={(w) => (secondaryWidth = w)}
		/>
	{/if}
	{#if datasetSidebarOpen && $activeDataset}
		<DatasetSidebar
			dataset={$activeDataset}
			onClose={() => setActiveDatasetId(null)}
			onWidthChange={(w) => (secondaryWidth = w)}
		/>
	{/if}
</div>

<style>
	.content {
		flex: 1;
		display: flex;
		flex-direction: column;
		min-width: 0;
		overflow: auto;
	}
	.layout {
		display: flex;
		flex: 1;
		min-height: 0;
		height: calc(100vh - var(--header-height));
		overflow: hidden;
		margin-top: var(--header-height);
	}
</style>
