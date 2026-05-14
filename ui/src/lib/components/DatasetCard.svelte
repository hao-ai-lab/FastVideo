<script lang="ts">
	import type { Dataset } from "$lib/api";
	import { deleteDataset } from "$lib/api";
	import { activeDatasetId, setActiveDatasetId } from "$lib/stores/activeDataset";

	let { dataset, onUpdated, onSelect = () => {} }: {
		dataset: Dataset;
		onUpdated: () => void;
		onSelect?: () => void;
	} = $props();

	let isLoading = $state(false);

	const isSelected = $derived($activeDatasetId === dataset.id);

	const fileCount = $derived(dataset.file_count ?? 0);
	const sizeBytes = $derived(dataset.size_bytes ?? 0);
	const sizeLabel = $derived(
		sizeBytes < 1024
			? `${sizeBytes} B`
			: sizeBytes < 1024 * 1024
				? `${(sizeBytes / 1024).toFixed(1)} KB`
				: sizeBytes < 1024 * 1024 * 1024
					? `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`
					: `${(sizeBytes / (1024 * 1024 * 1024)).toFixed(1)} GB`,
	);

	async function handleDelete(e: MouseEvent) {
		e.preventDefault();
		e.stopPropagation();
		if (isLoading) return;
		if (!confirm(`Delete dataset "${dataset.name}"?`)) return;
		isLoading = true;
		try {
			await deleteDataset(dataset.id);
			if ($activeDatasetId === dataset.id) setActiveDatasetId(null);
			onUpdated();
		} catch (err) {
			alert(
				err instanceof Error ? err.message : "Failed to delete dataset",
			);
		} finally {
			isLoading = false;
		}
	}

	function handleKeyDown(e: KeyboardEvent) {
		if (e.key === "Enter" || e.key === " ") {
			e.preventDefault();
			onSelect();
		}
	}
</script>

<div
	class="jobCard"
	class:jobCardSelected={isSelected}
	onclick={(e) => {
		if ((e.target as HTMLElement).closest("button")) return;
		onSelect();
	}}
	onkeydown={handleKeyDown}
	role="button"
	tabindex="0"
>
	<div class="jobHeader">
		<span class="jobModel">{dataset.name}</span>
		<button
			type="button"
			class="btn btnDelete btnSmall"
			onclick={handleDelete}
			disabled={isLoading}
		>
			Delete
		</button>
	</div>
	<div class="jobPrompt">
		{fileCount} {fileCount === 1 ? "file" : "files"} · {sizeLabel}
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
	}
	.btn {
		cursor: pointer;
		border: none;
		border-radius: var(--radius);
		font-family: inherit;
		font-weight: 600;
		font-size: 0.85rem;
		padding: 0.55rem 1.2rem;
	}
	.btnSmall {
		background: transparent;
		color: var(--accent-h);
		border: 1px solid var(--border);
		padding: 0.3rem 0.7rem;
		font-size: 0.75rem;
	}
	.btnDelete {
		background: var(--red);
		color: #fff;
	}
</style>
