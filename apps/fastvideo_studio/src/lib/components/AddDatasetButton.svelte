<script lang="ts">
	import { createDatasetModalOpen } from "$lib/stores/createDatasetModalOpen";

	function openModal() {
		createDatasetModalOpen.set(true);
	}
</script>

<button type="button" class="btn btnPrimary" onclick={openModal}>
	Add Dataset
</button>

<style>
	.btn {
		cursor: pointer;
		border: none;
		border-radius: var(--radius);
		font-family: inherit;
		font-weight: 600;
		font-size: 0.85rem;
		padding: 0.55rem 1.2rem;
	}
	.btnPrimary {
		background: var(--accent);
		color: #fff;
	}
</style>
