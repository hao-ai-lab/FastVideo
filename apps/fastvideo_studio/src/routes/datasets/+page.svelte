<script lang="ts">
	import { onMount, onDestroy } from "svelte";
	import { getDatasets } from "$lib/api";
	import type { Dataset } from "$lib/api";
	import CreateDatasetModal from "$lib/components/CreateDatasetModal.svelte";
	import DatasetCard from "$lib/components/DatasetCard.svelte";
	import AddDatasetButton from "$lib/components/AddDatasetButton.svelte";
	import { setHeaderActions } from "$lib/stores/headerActions";
	import {
		activeDataset,
		setActiveDataset,
		setActiveDatasetId,
	} from "$lib/stores/activeDataset";
	import { createDatasetModalOpen } from "$lib/stores/createDatasetModalOpen";

	let datasets = $state<Dataset[]>([]);

	onMount(() => {
		setHeaderActions([{ component: AddDatasetButton, props: {} }]);
	});
	onDestroy(() => setHeaderActions([]));

	async function fetchDatasets() {
		try {
			datasets = await getDatasets();
		} catch (err) {
			console.error("Failed to fetch datasets:", err);
		}
	}

	onMount(() => {
		fetchDatasets();
	});

	function handleSelectDataset(ds: Dataset) {
		setActiveDataset(ds);
		setActiveDatasetId(ds.id);
	}

	function handleCloseModal() {
		createDatasetModalOpen.set(false);
	}

	function handleSuccess() {
		fetchDatasets();
		createDatasetModalOpen.set(false);
	}
</script>

<main class="main">
	<section class="card">
		<div id="datasets-container">
			{#if datasets.length === 0}
				<p class="placeholder">No datasets yet.</p>
			{:else}
				{#each datasets as ds (ds.id)}
					<DatasetCard
						dataset={ds}
						onUpdated={fetchDatasets}
						onSelect={() => handleSelectDataset(ds)}
					/>
				{/each}
			{/if}
		</div>
	</section>
</main>
<CreateDatasetModal
	isOpen={$createDatasetModalOpen}
	onClose={handleCloseModal}
	onSuccess={handleSuccess}
/>

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
